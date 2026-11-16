"""OSM import helpers (no SUMO, no internet) and the committed Koteshwor network."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tlcs.network import load_network_spec
from tlcs.osm_import import (
    BBox,
    Restriction,
    bearing_deg,
    check_osm_bounds,
    clip_osm,
    compass,
    connection_deletions_xml,
    group_by_bearing,
    haversine_m,
    load_import_config,
    mean_bearing,
    netconvert_osm_cmd,
    netconvert_tls_cmd,
    osm_way,
    per_approach_program,
    read_restrictions,
    restriction_deletions,
)

KOTESHWOR = BBox(south=27.677, west=85.347, north=27.681, east=85.351)

# --- bounding box ----------------------------------------------------------------------------


def test_bbox_argument_order_is_lon_lat() -> None:
    assert KOTESHWOR.lon_lat_corners() == "85.347,27.677,85.351,27.681"


def test_bbox_rejects_swapped_coordinates() -> None:
    with pytest.raises(ValueError, match="latitudes"):
        BBox(south=27.681, west=85.347, north=27.677, east=85.351)
    with pytest.raises(ValueError, match="longitudes"):
        BBox(south=27.677, west=85.351, north=27.681, east=85.347)


def test_swapped_lat_lon_is_caught_by_the_junction_check(tmp_path: Path) -> None:
    # 85.3 is a valid latitude, so the ranges alone cannot catch a lat/lon swap; the expected
    # junction then falls outside the box, which load_import_config rejects.
    swapped = tmp_path / "bbox.yaml"
    swapped.write_text(
        "bbox: {south: 85.347, west: 27.677, north: 85.351, east: 27.681}\n"
        "junction: {lat: 27.6786, lon: 85.3494}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the box"):
        load_import_config(swapped)


def test_bbox_contains_and_matches() -> None:
    assert KOTESHWOR.contains(27.6786, 85.3494)
    assert not KOTESHWOR.contains(85.3494, 27.6786)
    assert KOTESHWOR.matches(BBox(27.67705, 85.34705, 27.68095, 85.35095))
    assert not KOTESHWOR.matches(BBox(27.6772, 85.347, 27.681, 85.351))


def test_committed_koteshwor_config() -> None:
    config = load_import_config(Path("intersection/koteshwor/bbox.yaml"))
    assert config.bbox == KOTESHWOR
    assert (config.junction_lat, config.junction_lon) == (27.6786, 85.3494)


# --- OSM file: bounds and clipping -----------------------------------------------------------

OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
 <bounds minlat="0" minlon="0" maxlat="1" maxlon="1"/>
 <node id="1" lat="0.5" lon="-0.5"/>
 <node id="2" lat="0.5" lon="0.5"/>
 <node id="3" lat="0.5" lon="1.5"/>
 <node id="4" lat="2" lon="2"/>
 <node id="5" lat="3" lon="3"/>
 <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><tag k="highway" v="trunk"/></way>
 <way id="11"><nd ref="4"/><nd ref="5"/><tag k="highway" v="trunk"/></way>
 <relation id="20">
  <member type="way" ref="10" role="from"/><member type="node" ref="2" role="via"/>
  <member type="way" ref="12" role="to"/>
  <tag k="type" v="restriction"/><tag k="restriction" v="no_left_turn"/>
 </relation>
</osm>
"""


def _write(tmp_path: Path, text: str = OSM) -> Path:
    path = tmp_path / "in.osm"
    path.write_text(text, encoding="utf-8")
    return path


def test_check_osm_bounds(tmp_path: Path) -> None:
    path = _write(tmp_path)
    check_osm_bounds(path, BBox(0, 0, 1, 1))
    with pytest.raises(ValueError, match="do not match"):
        check_osm_bounds(path, BBox(0, 0, 1, 1.01))


def test_clip_osm_cuts_ways_at_the_box_edge(tmp_path: Path) -> None:
    out = tmp_path / "out.osm"
    clip_osm(_write(tmp_path), out, BBox(0, 0, 1, 1))
    root = ET.parse(out).getroot()  # noqa: S314
    nodes = {
        n.get("id"): (float(n.get("lat", "")), float(n.get("lon", ""))) for n in root.iter("node")
    }
    ways = {w.get("id"): [nd.get("ref") for nd in w.iter("nd")] for w in root.iter("way")}

    assert set(ways) == {"10"}  # way 11 lies fully outside: removed
    refs = ways["10"]
    assert refs[1] == "2"  # the inside node is kept
    assert nodes[refs[0]] == pytest.approx((0.5, 0.0))  # new node on the west edge
    assert nodes[refs[2]] == pytest.approx((0.5, 1.0))  # new node on the east edge
    assert all(0 <= lat <= 1 and 0 <= lon <= 1 for lat, lon in nodes.values())


def test_clip_osm_splits_a_way_that_leaves_and_reenters(tmp_path: Path) -> None:
    text = OSM.replace(
        '<way id="11"><nd ref="4"/><nd ref="5"/>',
        '<node id="6" lat="0.2" lon="0.5"/><way id="11"><nd ref="2"/><nd ref="3"/><nd ref="6"/>',
    )
    out = tmp_path / "out.osm"
    clip_osm(_write(tmp_path, text), out, BBox(0, 0, 1, 1))
    ways = list(ET.parse(out).getroot().iter("way"))  # noqa: S314
    ids = sorted(w.get("id", "") for w in ways)
    assert "11" in ids
    assert any(i.startswith("-") for i in ids)  # the re-entering part is a new (negative) way


# --- netconvert commands -----------------------------------------------------------------------


def test_import_command_is_left_hand_cars_only() -> None:
    cmd = netconvert_osm_cmd(Path("a.osm"), Path("b.net.xml"), join_dist=15.0)
    for flag in ("--lefthand", "--remove-edges.isolated", "--junctions.join", "--no-turnarounds"):
        assert flag in cmd
    assert cmd[cmd.index("--junctions.join-dist") + 1] == "15.0"
    types = cmd[cmd.index("--keep-edges.by-type") + 1].split(",")
    assert "highway.trunk" in types
    assert not {"highway.footway", "highway.cycleway", "highway.path"} & set(types)


def test_tls_command() -> None:
    forced = netconvert_tls_cmd(Path("a.net.xml"), Path("b.net.xml"), "J1", ["J2"], Path("c.xml"))
    assert forced[forced.index("--tls.set") + 1] == "J1"
    assert forced[forced.index("--tls.unset") + 1] == "J2"
    assert "--lefthand" in forced
    kept = netconvert_tls_cmd(Path("a.net.xml"), Path("b.net.xml"), None)
    assert "--tls.set" not in kept


# --- turn restrictions -------------------------------------------------------------------------


def test_read_restrictions(tmp_path: Path) -> None:
    assert read_restrictions(_write(tmp_path)) == [Restriction("no_left_turn", "10", "12")]


def test_restriction_deletions() -> None:
    connections = [("10#1", "12#0"), ("10#1", "13"), ("-11", "12#0"), ("-11", "14")]
    assert osm_way("-11#3") == "11"
    assert restriction_deletions([Restriction("no_left_turn", "10", "12")], connections) == [
        ("10#1", "12#0")
    ]
    assert restriction_deletions([Restriction("only_straight_on", "11", "14")], connections) == [
        ("-11", "12#0")
    ]
    xml = connection_deletions_xml([("10#1", "12#0")])
    assert ET.fromstring(xml).find("delete").attrib == {"from": "10#1", "to": "12#0"}  # noqa: S314


# --- geometry ----------------------------------------------------------------------------------


def test_geometry_helpers() -> None:
    assert haversine_m(27.6786, 85.3494, 27.6786, 85.3494) == 0
    assert haversine_m(27.677, 85.349, 27.681, 85.349) == pytest.approx(444.8, abs=0.5)
    assert bearing_deg(0, 1) == 0
    assert bearing_deg(1, 0) == 90
    assert compass(6) == "N"
    assert compass(244) == "SW"
    assert mean_bearing([356, 16]) == pytest.approx(6)  # not 186


def test_group_by_bearing() -> None:
    # Koteshwor: N arm (356, 17), S arm (165, 183), SW arm (239, 244, 250).
    groups = group_by_bearing([17, 183, 250, 244, 356, 165, 239])
    assert groups == [[0, 4], [5, 1], [6, 3, 2]]


# --- phase plan ----------------------------------------------------------------------------------


def test_per_approach_program() -> None:
    xml = per_approach_program("J", link_arms=[0, 0, 2, 1, 2], n_arms=3, yellow=4)
    phases = [(p.get("state"), p.get("duration")) for p in ET.fromstring(xml).iter("phase")]  # noqa: S314
    assert phases == [
        ("GGrrr", "30"), ("yyrrr", "4"),
        ("rrrGr", "30"), ("rrryr", "4"),
        ("rrGrG", "30"), ("rryry", "4"),
    ]  # fmt: skip


# --- the committed Koteshwor network --------------------------------------------------------------


def test_koteshwor_network() -> None:
    net = ET.parse("intersection/koteshwor/environment.net.xml").getroot()  # noqa: S314
    assert net.get("lefthand") == "true"
    spec = load_network_spec("koteshwor", signal_program="phases_per_approach.add.xml")
    assert spec.num_actions == 3  # one green per approach
    assert {osm_way(e) for e in spec.incoming_edges} == {
        "193289327",  # N: Kathmandu Ring Road
        "226834863",  # S: Araniko Highway
        "648697035", "1126627065",  # SW: Kathmandu Ring Road (two roadways)
    }  # fmt: skip
    # OSM no_left_turn from Araniko Highway (226834863) into the Ring Road (215507255).
    moves = {(osm_way(r.in_edge), osm_way(r.out_edge)) for r in spec.routes}
    assert ("226834863", "215507255") not in moves
    # Traffic is generated only for real movements (no U-turns).
    assert [r for r in spec.routes if r.is_uturn]
    # Routes start and end at the edge of the box (e.g. Araniko starts before the slip lane).
    araniko = next(r for r in spec.routes if osm_way(r.in_edge) == "226834863")
    assert araniko.path[0] == "226834863#0"

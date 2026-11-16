"""Import a signalized junction from OpenStreetMap into intersection/<network>/.

Usage (from the project root):

    uv run python scripts/import_osm.py                      # download, build, verify
    uv run python scripts/import_osm.py --osm-file my.osm    # use a local export (same box)
    uv run python scripts/import_osm.py --verify-only        # re-check after netedit edits

The network folder must contain bbox.yaml:

    bbox:     {south: ..., west: ..., north: ..., east: ...}
    junction: {lat: ..., lon: ...}

Build steps: download (or local file) -> check bounds -> clip ways to the box -> netconvert
(left-hand, cars-only arterials, isolated edges removed, OSM one-ways and turn restrictions kept)
with the junction's OSM nodes joined into one junction -> traffic light forced on it ->
environment.net.xml, sumo_config.sumocfg and a one-green-per-approach signal plan.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import sumolib

from tlcs.network import NETWORKS_DIR, follow_road
from tlcs.osm_import import (
    ImportConfig,
    bearing_deg,
    check_osm_bounds,
    clip_osm,
    compass,
    connection_deletions_xml,
    download_osm,
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

NET_NAME = "environment.net.xml"
RAW_OSM_NAME = "map.osm"
PROGRAM_NAME = "phases_per_approach.add.xml"
PNG_NAME = "network.png"
SUMOCFG = """<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net}" />
        <route-files value="episode_routes.rou.xml" />
    </input>

    <time>
        <begin value="0" />
    </time>

    <processing>
        <time-to-teleport value="-1" />
    </processing>

</configuration>
"""
JUNCTION_TOLERANCE_M = 50.0
MIN_ARM_LENGTH_M = 150.0
BOUNDARY_TOLERANCE_M = 25.0  # lane widths make the network a few meters wider than its axes
ARM_PROBE_M = 30.0  # bearing of an arm is measured this far from the junction


def run(cmd: list[str]) -> None:
    """Run netconvert and stop with its output if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")


def nearest_node(net: sumolib.net.Net, lat: float, lon: float) -> sumolib.net.node.Node:
    """Network node closest to a lat/lon point (fringe dead ends excluded)."""
    x, y = net.convertLonLat2XY(lon, lat)
    nodes = [n for n in net.getNodes() if n.getIncoming() and n.getOutgoing()]
    return min(nodes, key=lambda n: (n.getCoord()[0] - x) ** 2 + (n.getCoord()[1] - y) ** 2)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(config: ImportConfig, folder: Path, osm_file: Path | None, join_dist: float) -> None:
    """Download/check/clip the OSM data and build the SUMO network in `folder`."""
    raw = folder / RAW_OSM_NAME
    if osm_file is not None:
        if osm_file.resolve() != raw.resolve():
            shutil.copyfile(osm_file, raw)
        print(f"Using local OSM file {osm_file} (stored as {raw})")
    else:
        print(f"Downloading OSM data for bbox {config.bbox.lon_lat_corners()} (lon/lat corners)")
        download_osm(config.bbox, raw)
    bounds = check_osm_bounds(raw, config.bbox)
    print(
        f"OSM bounds match the configured box: south={bounds.south} west={bounds.west} "
        f"north={bounds.north} east={bounds.east}"
    )

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        clipped = tmp / "clipped.osm"
        clip_osm(raw, clipped, config.bbox)

        # Pass 1: import; nodes linked by edges shorter than join_dist become one junction.
        joined = tmp / "joined.net.xml"
        run(netconvert_osm_cmd(clipped, joined, join_dist=join_dist))
        net = sumolib.net.readNet(str(joined), withPrograms=True)
        junction = nearest_node(net, config.junction_lat, config.junction_lon)
        # The traffic light(s) controlling the junction: identified by the links they control.
        own = {
            t.getID()
            for t in net.getTrafficLights()
            if any(c[0].getEdge().getToNode() is junction for c in t.getConnections())
        }
        others = [t.getID() for t in net.getTrafficLights() if t.getID() not in own]
        print(f"Junction: '{junction.getID()}' ({len(junction.getIncoming())} incoming edges)")

        # Re-apply OSM turn restrictions: joining removes the via-nodes netconvert used for them.
        connections = sorted(
            {
                (edge.getID(), target.getID())
                for edge in junction.getIncoming()
                for target in edge.getOutgoing()
                if target in junction.getOutgoing()
            }
        )
        deletions = restriction_deletions(read_restrictions(clipped), connections)
        connection_file = tmp / "restrictions.con.xml"
        connection_file.write_text(connection_deletions_xml(deletions), encoding="utf-8")
        for src, dst in deletions:
            print(f"Turn restriction applied: removed {src} -> {dst}")

        # Pass 2: force the traffic light on the junction, remove any other traffic light.
        run(
            netconvert_tls_cmd(
                joined,
                folder / NET_NAME,
                None if own else junction.getID(),
                others,
                connection_file,
            )
        )
        how = f"kept OSM traffic light {sorted(own)}" if own else "traffic light forced"
        print(f"Junction '{junction.getID()}': {how}" + (f"; removed {others}" if others else ""))

    (folder / "sumo_config.sumocfg").write_text(SUMOCFG.format(net=NET_NAME), encoding="utf-8")
    print(f"Wrote {folder / NET_NAME} and {folder / 'sumo_config.sumocfg'}")


# ---------------------------------------------------------------------------
# Verify and report
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    """One leg of the junction: its edges in both directions."""

    name: str
    bearing: float
    incoming: list[sumolib.net.edge.Edge]
    outgoing: list[sumolib.net.edge.Edge]


def _point_along(edge: sumolib.net.edge.Edge, from_junction: bool) -> tuple[float, float]:
    """Point about ARM_PROBE_M along an edge, measured from the junction end."""
    shape = edge.getShape() if from_junction else list(reversed(edge.getShape()))
    remaining = ARM_PROBE_M
    for (x1, y1), (x2, y2) in pairwise(shape):
        seg = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if seg >= remaining:
            f = remaining / seg
            return x1 + f * (x2 - x1), y1 + f * (y2 - y1)
        remaining -= seg
    return shape[-1]


def find_arms(junction: sumolib.net.node.Node) -> list[Arm]:
    """Group the junction's edges into arms by compass direction."""
    jx, jy = junction.getCoord()
    edges = [(e, True) for e in junction.getIncoming()] + [
        (e, False) for e in junction.getOutgoing()
    ]
    bearings = []
    for edge, incoming in edges:
        px, py = _point_along(edge, from_junction=not incoming)
        bearings.append(bearing_deg(px - jx, py - jy))
    arms = []
    for group in group_by_bearing(bearings):
        members = [edges[i] for i in group]
        mean = mean_bearing([bearings[i] for i in group])
        names = Counter(e.getName() for e, _ in members if e.getName())
        road = names.most_common(1)[0][0] if names else "unnamed road"
        arms.append(
            Arm(
                name=f"{compass(mean)} ({road})",
                bearing=mean,
                incoming=[e for e, inc in members if inc],
                outgoing=[e for e, inc in members if not inc],
            )
        )
    return arms


def verify(config: ImportConfig, folder: Path, write_program: bool) -> None:  # noqa: C901, PLR0912, PLR0915
    """Coordinate checks, network summary, arm lengths, plot and (optionally) the phase plan."""
    net_file = folder / NET_NAME
    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    lights = net.getTrafficLights()
    if len(lights) != 1:
        print(
            f"WARNING: expected 1 traffic light, found {len(lights)}: {[t.getID() for t in lights]}"
        )
    tls = lights[0]
    junction = next(iter({c[0].getEdge().getToNode() for c in tls.getConnections()}))

    # --- 2. coordinates -------------------------------------------------------------------
    print("\n== Coordinate verification")
    lon, lat = net.convertXY2LonLat(*junction.getCoord())
    dist = haversine_m(lat, lon, config.junction_lat, config.junction_lon)
    inside = config.bbox.contains(lat, lon)
    print(f"Controlled junction '{junction.getID()}' at lat {lat:.6f}, lon {lon:.6f}")
    print(f"  inside bbox: {'yes' if inside else 'NO'}")
    print(
        f"  distance to expected point ({config.junction_lat}, {config.junction_lon}): {dist:.1f} m"
    )
    if not inside or dist > JUNCTION_TOLERANCE_M:
        print(
            f"WARNING: junction is more than {JUNCTION_TOLERANCE_M:.0f} m from the expected point "
            "or outside the box - the wrong junction may have been chosen!"
        )
    points = [p for e in net.getEdges() for ln in e.getLanes() for p in ln.getShape()]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    west, south = net.convertXY2LonLat(min(xs), min(ys))
    east, north = net.convertXY2LonLat(max(xs), max(ys))
    print(f"Network boundary: south={south:.6f} west={west:.6f} north={north:.6f} east={east:.6f}")
    box = config.bbox
    # (distance network edge <-> box edge, True if the network goes beyond the box)
    sides = {
        "south": (haversine_m(south, lon, box.south, lon), south < box.south),
        "north": (haversine_m(north, lon, box.north, lon), north > box.north),
        "west": (haversine_m(lat, west, lat, box.west), west < box.west),
        "east": (haversine_m(lat, east, lat, box.east), east > box.east),
    }
    for side, (dist_m, beyond) in sides.items():
        if dist_m <= BOUNDARY_TOLERANCE_M:
            status = "matches the box edge"
        elif beyond:
            status = "MISMATCH: network extends beyond the box"
        else:
            status = "inside the box (no kept road reaches this edge)"
        print(f"  {side:5s}: {dist_m:6.1f} m {'outside' if beyond else 'inside'} - {status}")

    # --- 3. summary -----------------------------------------------------------------------
    arms = find_arms(junction)
    arm_of_edge = {e.getID(): i for i, arm in enumerate(arms) for e in arm.incoming + arm.outgoing}
    print(f"\n== Network summary\nJunction: {junction.getID()} (traffic light '{tls.getID()}')")
    print(f"Arms: {len(arms)} (expected 3)")
    for arm in arms:
        print(f"  {arm.name}, bearing {arm.bearing:.0f} deg")
        for edge in arm.incoming:
            lanes = ", ".join(f"{ln.getID()} {ln.getLength():.0f} m" for ln in edge.getLanes())
            print(f"    in : {edge.getID()}  [{lanes}]")
        for edge in arm.outgoing:
            print(f"    out: {edge.getID()}  ({edge.getLaneNumber()} lanes)")

    links = sorted(tls.getConnections(), key=lambda c: c[2])
    print("Allowed turning movements (through the traffic light):")
    moves: dict[tuple[int, int, str], list[str]] = {}
    for in_lane, out_lane, index in links:
        direction = next(
            c.getDirection() for c in in_lane.getOutgoing() if c.getToLane() is out_lane
        )
        key = (
            arm_of_edge[in_lane.getEdge().getID()],
            arm_of_edge[out_lane.getEdge().getID()],
            direction,
        )
        moves.setdefault(key, []).append(f"{in_lane.getID()}->{out_lane.getID()} (link {index})")
    names = {
        "s": "straight",
        "l": "left",
        "r": "right",
        "L": "partial left",
        "R": "partial right",
        "t": "U-turn",
    }
    for (a, b, d), lanes in sorted(moves.items()):
        print(f"  {arms[a].name} -> {arms[b].name}: {names.get(d, d)}; {'; '.join(lanes)}")

    raw = folder / RAW_OSM_NAME
    if raw.exists():
        print("OSM turn restrictions at this junction:")
        junction_ways = {
            osm_way(e.getID()) for e in junction.getIncoming() + junction.getOutgoing()
        }
        through = sorted({(i.getEdge().getID(), o.getEdge().getID()) for i, o, _ in links})
        for r in read_restrictions(raw):
            if r.from_way not in junction_ways or r.to_way not in junction_ways:
                continue
            violated = restriction_deletions([r], through)
            status = "respected" if not violated else f"VIOLATED by {violated}"
            print(f"  {r.kind} from way {r.from_way} to way {r.to_way}: {status}")

    print("Signal program in the net file:")
    for program in tls.getPrograms().values():
        for i, phase in enumerate(program.getPhases()):
            print(f"  phase {i}: {phase.state}  ({phase.duration:.0f} s)")

    # --- 4. arm lengths -------------------------------------------------------------------
    print("\n== Arm length (junction -> box edge, along the road)")
    junction_edges = {e.getID() for e in junction.getIncoming() + junction.getOutgoing()}
    for arm in arms:
        for edge in arm.incoming:
            upstream = follow_road(edge, forward=False, avoid=junction_edges)
            length = edge.getLength() + sum(net.getEdge(e).getLength() for e in upstream)
            warn = (
                "  WARNING: short - queues may reach the network edge at high demand"
                if length < MIN_ARM_LENGTH_M
                else ""
            )
            print(f"  {arm.name}: {length:.0f} m via {[*reversed(upstream), edge.getID()]}{warn}")

    plot(net, config, junction, arms, folder / PNG_NAME, (lat, lon))

    # --- 6. phase plan --------------------------------------------------------------------
    program_file = folder / PROGRAM_NAME
    link_arms = [arm_of_edge[in_lane.getEdge().getID()] for in_lane, _, _ in links]
    if write_program:
        program_file.write_text(
            per_approach_program(tls.getID(), link_arms, len(arms)), encoding="utf-8"
        )
        print(f"\nWrote one-green-per-approach signal plan: {program_file}")
    if program_file.exists():
        root = ET.parse(program_file).getroot()  # noqa: S314
        states = [p.get("state", "") for p in root.iter("phase")]
        print(f"Signal plan {program_file.name}:")
        for i, state in enumerate(states):
            print(f"  phase {i}: {state}")
        if any(len(s) != len(links) for s in states):
            print(
                f"WARNING: plan states must have {len(links)} characters; re-run with --write-program"
            )


def plot(  # noqa: PLR0913
    net: sumolib.net.Net,
    config: ImportConfig,
    junction: sumolib.net.node.Node,
    arms: list[Arm],
    out_file: Path,
    junction_lat_lon: tuple[float, float],
) -> None:
    """PNG of the network in lat/lon with the bounding box and the junction labelled."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(9, 9))
    for edge in net.getEdges():
        for lane in edge.getLanes():
            pts = [net.convertXY2LonLat(x, y) for x, y in lane.getShape()]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#55544f", linewidth=0.8)
    box = config.bbox
    ax.plot(
        [box.west, box.east, box.east, box.west, box.west],
        [box.south, box.south, box.north, box.north, box.south],
        color="#2a78d6", linestyle="--", linewidth=1.5, label="bounding box",
    )  # fmt: skip
    lat, lon = junction_lat_lon
    ax.plot(lon, lat, "o", color="#e34948", markersize=10, label="controlled junction")
    ax.annotate(
        f"{lat:.5f} N, {lon:.5f} E", (lon, lat), xytext=(10, -18), textcoords="offset points"
    )
    ax.plot(
        config.junction_lon,
        config.junction_lat,
        "x",
        color="#2b2a27",
        markersize=10,
        label="expected point",
    )
    jx, jy = junction.getCoord()
    for arm in arms:
        ends = [p for e in arm.incoming + arm.outgoing for p in (e.getShape()[0], e.getShape()[-1])]
        far = max(ends, key=lambda p: (p[0] - jx) ** 2 + (p[1] - jy) ** 2)
        alon, alat = net.convertXY2LonLat(*far)
        ax.annotate(
            arm.name,
            (alon, alat),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=9,
            color="#1c5cab",
        )
    for side, value in (("S", box.south), ("N", box.north)):
        ax.annotate(f"{side} {value}", (box.west, value), fontsize=8, color="#2a78d6")
    for side, value in (("W", box.west), ("E", box.east)):
        ax.annotate(f"{side} {value}", (value, box.south), fontsize=8, color="#2a78d6", rotation=90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{out_file.parent.name}: imported network")
    ax.set_aspect(1 / abs(__import__("math").cos(__import__("math").radians(lat))))
    ax.legend(loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    print(f"\nPlot: {out_file}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--network", default="koteshwor", help="Folder under intersection/ (default koteshwor)"
    )
    parser.add_argument(
        "--osm-file", type=Path, help="Local .osm export of the same box (skips the download)"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only re-run the checks on the existing network"
    )
    parser.add_argument(
        "--write-program", action="store_true", help="With --verify-only: rewrite the signal plan"
    )
    parser.add_argument(
        "--join-dist",
        type=float,
        default=15.0,
        help="Nodes linked by edges shorter than this (m) are joined (default 15)",
    )
    args = parser.parse_args()

    folder = NETWORKS_DIR / args.network
    config = load_import_config(folder / "bbox.yaml")
    if not args.verify_only:
        build(config, folder, args.osm_file, args.join_dist)
    verify(config, folder, write_program=args.write_program or not args.verify_only)


if __name__ == "__main__":
    main()

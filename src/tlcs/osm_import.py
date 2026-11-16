"""Import a single signalized junction from OpenStreetMap into a SUMO network.

Used by `scripts/import_osm.py`. The pure helpers (bounding box handling, OSM clipping, arm
grouping, phase plan building) are kept here so they can be tested without SUMO or internet.
"""

import math
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map?bbox={bbox}"
USER_AGENT = "tlcs-osm-import/0.1"
EARTH_RADIUS_M = 6_371_000.0
MAX_LAT, MAX_LON = 90.0, 180.0
MIN_WAY_NODES = 2

# Road types kept for a cars-only arterial network (footways, cycleways, paths, steps, service
# roads and residential lanes are dropped).
DEFAULT_ROAD_TYPES = (
    "highway.trunk",
    "highway.trunk_link",
    "highway.primary",
    "highway.primary_link",
    "highway.secondary",
    "highway.secondary_link",
    "highway.tertiary",
    "highway.tertiary_link",
)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """A latitude/longitude box. Every tool's argument order is built from the named fields."""

    south: float  # min latitude
    west: float  # min longitude
    north: float  # max latitude
    east: float  # max longitude

    def __post_init__(self) -> None:
        """Reject boxes with swapped or out-of-range coordinates."""
        if not (-MAX_LAT <= self.south < self.north <= MAX_LAT):
            msg = f"Invalid latitudes: south={self.south} must be < north={self.north}"
            raise ValueError(msg)
        if not (-MAX_LON <= self.west < self.east <= MAX_LON):
            msg = f"Invalid longitudes: west={self.west} must be < east={self.east}"
            raise ValueError(msg)

    def lon_lat_corners(self) -> str:
        """Return "min_lon,min_lat,max_lon,max_lat" (left,bottom,right,top) for OSM/netconvert."""
        return f"{self.west},{self.south},{self.east},{self.north}"

    def contains(self, lat: float, lon: float) -> bool:
        """True if the point lies inside the box (edges included)."""
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def matches(self, other: "BBox", tolerance: float = 1e-4) -> bool:
        """True if every side differs by at most `tolerance` degrees."""
        return all(
            abs(getattr(self, side) - getattr(other, side)) <= tolerance
            for side in ("south", "west", "north", "east")
        )


@dataclass(frozen=True)
class ImportConfig:
    """Contents of a network's bbox.yaml."""

    bbox: BBox
    junction_lat: float
    junction_lon: float


def load_import_config(path: Path) -> ImportConfig:
    """Read bbox.yaml: `bbox: {south, west, north, east}` and `junction: {lat, lon}`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    box = data["bbox"]
    junction = data["junction"]
    config = ImportConfig(
        bbox=BBox(
            south=float(box["south"]),
            west=float(box["west"]),
            north=float(box["north"]),
            east=float(box["east"]),
        ),
        junction_lat=float(junction["lat"]),
        junction_lon=float(junction["lon"]),
    )
    if not config.bbox.contains(config.junction_lat, config.junction_lon):
        msg = f"Junction ({config.junction_lat}, {config.junction_lon}) is outside the box"
        raise ValueError(msg)
    return config


# ---------------------------------------------------------------------------
# OSM data: download, bounds check, clipping
# ---------------------------------------------------------------------------


def download_osm(bbox: BBox, out_file: Path, timeout: float = 120.0) -> None:
    """Download all OSM data in the box from the OSM API (only the box is sent)."""
    url = OSM_API_URL.format(bbox=bbox.lon_lat_corners())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        out_file.write_bytes(response.read())


def read_osm_bounds(osm_file: Path) -> BBox:
    """Return the <bounds> element of an .osm file as a BBox."""
    for _, element in ET.iterparse(osm_file, events=("start",)):  # noqa: S314
        if element.tag == "bounds":
            return BBox(
                south=float(element.get("minlat", "nan")),
                west=float(element.get("minlon", "nan")),
                north=float(element.get("maxlat", "nan")),
                east=float(element.get("maxlon", "nan")),
            )
    msg = f"No <bounds> element in '{osm_file}'"
    raise ValueError(msg)


def check_osm_bounds(osm_file: Path, bbox: BBox, tolerance: float = 1e-4) -> BBox:
    """Stop with a clear error if the file's bounds differ from the configured box."""
    bounds = read_osm_bounds(osm_file)
    if not bounds.matches(bbox, tolerance):
        msg = (
            f"Bounds of '{osm_file}' do not match the configured box (tolerance {tolerance} deg):\n"
            f"  file:       south={bounds.south} west={bounds.west} "
            f"north={bounds.north} east={bounds.east}\n"
            f"  configured: south={bbox.south} west={bbox.west} "
            f"north={bbox.north} east={bbox.east}"
        )
        raise ValueError(msg)
    return bounds


def _crossing(
    inside: tuple[float, float], outside: tuple[float, float], bbox: BBox
) -> tuple[float, float]:
    """Point where the segment inside->outside (lat, lon) leaves the box."""
    (lat1, lon1), (lat2, lon2) = inside, outside
    t = 1.0
    for value, low, high, start, end in (
        (lat2, bbox.south, bbox.north, lat1, lat2),
        (lon2, bbox.west, bbox.east, lon1, lon2),
    ):
        if value < low:
            t = min(t, (low - start) / (end - start))
        elif value > high:
            t = min(t, (high - start) / (end - start))
    return lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1)


def clip_way(
    refs: Sequence[str],
    coords: dict[str, tuple[float, float]],
    bbox: BBox,
    new_node: Callable[[float, float], str],
) -> list[list[str]]:
    """Cut one way's node list at the box edges.

    Args:
        refs: Node IDs of the way, in order.
        coords: (lat, lon) of every node.
        bbox: The box.
        new_node: Creates a node at (lat, lon) and returns its ID.

    Returns:
        The parts of the way inside the box (each with at least 2 nodes). A part that crosses the
        box edge ends (or starts) at a new node exactly on the edge.
    """
    parts: list[list[str]] = []
    current: list[str] = []
    prev: tuple[float, float] | None = None
    prev_inside = False
    for ref in refs:
        here = coords.get(ref)
        if here is None:
            continue
        inside = bbox.contains(*here)
        if inside:
            if prev is not None and not prev_inside:  # entering the box
                current = [new_node(*_crossing(here, prev, bbox))]
            current.append(ref)
        elif prev is not None and prev_inside:  # leaving the box
            current.append(new_node(*_crossing(prev, here, bbox)))
            parts.append(current)
            current = []
        prev, prev_inside = here, inside
    if current:
        parts.append(current)
    return [part for part in parts if len(part) >= MIN_WAY_NODES]


def _rewrite_way(
    root: ET.Element, way: ET.Element, parts: list[list[str]], new_id: Callable[[], str]
) -> None:
    """Replace a way's nodes with its clipped parts (extra parts become new ways)."""
    tags = way.findall("tag")
    for nd in way.findall("nd"):
        way.remove(nd)
    if not parts:
        root.remove(way)
        return
    for ref in parts[0]:
        ET.SubElement(way, "nd", {"ref": ref})
    for part in parts[1:]:
        extra = ET.SubElement(root, "way", {"id": new_id()})
        for ref in part:
            ET.SubElement(extra, "nd", {"ref": ref})
        for tag in tags:
            ET.SubElement(extra, "tag", tag.attrib)


def clip_osm(src: Path, dst: Path, bbox: BBox) -> None:
    """Cut every way at the box edges, so the network ends exactly at the box.

    Nodes outside the box are dropped. Where a way leaves (or enters) the box, a new node is
    created exactly on the box edge. A way that leaves and re-enters is split into parts (the
    first part keeps the way's ID so turn restrictions still refer to it; new objects get
    negative IDs, the OSM convention for new data).
    """
    tree = ET.parse(src)  # noqa: S314
    root = tree.getroot()
    coords: dict[str, tuple[float, float]] = {
        n.get("id", ""): (float(n.get("lat", "0")), float(n.get("lon", "0")))
        for n in root.iter("node")
    }
    new_nodes: list[ET.Element] = []
    next_id = -1

    def new_id() -> str:
        nonlocal next_id
        value = str(next_id)
        next_id -= 1
        return value

    def new_node(lat: float, lon: float) -> str:
        node_id = new_id()
        new_nodes.append(
            ET.Element("node", {"id": node_id, "lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
        )
        return node_id

    for way in list(root.iter("way")):
        parts = clip_way([nd.get("ref", "") for nd in way.findall("nd")], coords, bbox, new_node)
        _rewrite_way(root, way, parts, new_id)

    for node in list(root.iter("node")):
        if not bbox.contains(*coords[node.get("id", "")]):
            root.remove(node)
    first_way = next((i for i, el in enumerate(root) if el.tag == "way"), len(root))
    for offset, node in enumerate(new_nodes):
        root.insert(first_way + offset, node)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# netconvert
# ---------------------------------------------------------------------------


def netconvert_osm_cmd(
    osm_file: Path,
    out_file: Path,
    road_types: Sequence[str] = DEFAULT_ROAD_TYPES,
    join_dist: float = 15.0,
) -> list[str]:
    """Build the netconvert command importing OSM data as a left-hand, cars-only network.

    OSM one-way tags and turn restrictions are applied by netconvert's OSM importer; turn
    restrictions whose via-node disappears into a joined junction are re-applied afterwards
    (see `restriction_deletions`). `--osm.turn-lanes` imports turn arrows; no U-turns.

    Args:
        osm_file: Clipped OSM data.
        out_file: Network to write.
        road_types: OSM road types to keep.
        join_dist: Nodes linked by edges shorter than this (m) are joined into one junction.
    """
    return [
        "netconvert",
        "--osm-files", str(osm_file),
        "--lefthand",
        "--keep-edges.by-type", ",".join(road_types),
        "--remove-edges.isolated",
        "--geometry.remove",
        "--junctions.join",
        "--junctions.join-dist", str(join_dist),
        "--osm.turn-lanes",
        "--output.street-names",
        "--no-turnarounds",
        "--tls.default-type", "static",
        "-o", str(out_file),
    ]  # fmt: skip


def netconvert_tls_cmd(
    net_file: Path,
    out_file: Path,
    junction_id: str | None,
    unset_tls: Sequence[str] = (),
    connection_file: Path | None = None,
) -> list[str]:
    """Build the netconvert command setting a traffic light on one junction (unsetting others)."""
    cmd = [
        "netconvert",
        "--sumo-net-file", str(net_file),
        "--lefthand",
        "--no-turnarounds",
        "--tls.default-type", "static",
        "-o", str(out_file),
    ]  # fmt: skip
    if junction_id is not None:  # None: the junction already has a traffic light
        cmd += ["--tls.set", junction_id]
    if unset_tls:
        cmd += ["--tls.unset", ",".join(unset_tls)]
    if connection_file is not None:
        cmd += ["--connection-files", str(connection_file)]
    return cmd


@dataclass(frozen=True)
class Restriction:
    """An OSM turn restriction between two ways (e.g. no_left_turn, only_straight_on)."""

    kind: str
    from_way: str
    to_way: str


def read_restrictions(osm_file: Path) -> list[Restriction]:
    """Turn restriction relations of an .osm file."""
    found = []
    for rel in ET.parse(osm_file).getroot().iter("relation"):  # noqa: S314
        tags = {tag.get("k"): tag.get("v") for tag in rel.iter("tag")}
        if tags.get("type") != "restriction":
            continue
        roles = {m.get("role"): m.get("ref", "") for m in rel.iter("member")}
        found.append(
            Restriction(
                tags.get("restriction") or "", roles.get("from") or "", roles.get("to") or ""
            )
        )
    return found


def osm_way(edge_id: str) -> str:
    """OSM way ID an imported edge comes from ("-123#4" -> "123")."""
    return edge_id.lstrip("-").split("#")[0]


def restriction_deletions(
    restrictions: Sequence[Restriction],
    connections: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Edge connections through a junction that OSM turn restrictions forbid.

    Args:
        restrictions: OSM turn restrictions.
        connections: (incoming edge, outgoing edge) pairs connected through the junction.

    Returns:
        The (incoming edge, outgoing edge) pairs to delete: for "no_*" the banned turn, for
        "only_*" every other turn from that way.
    """
    banned = set()
    for r in restrictions:
        for src, dst in connections:
            if osm_way(src) != r.from_way:
                continue
            to_target = osm_way(dst) == r.to_way
            if (r.kind.startswith("no_") and to_target) or (
                r.kind.startswith("only_") and not to_target
            ):
                banned.add((src, dst))
    return sorted(banned)


def connection_deletions_xml(pairs: Sequence[tuple[str, str]]) -> str:
    """Return a netconvert connection file deleting the given edge connections."""
    rows = "".join(f'    <delete from="{a}" to="{b}"/>\n' for a, b in pairs)
    return f"<connections>\n{rows}</connections>\n"


# ---------------------------------------------------------------------------
# Geometry: distances, bearings, arms
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(dx: float, dy: float) -> float:
    """Compass bearing (0 = north, 90 = east) of a vector in SUMO x/y (x east, y north)."""
    return math.degrees(math.atan2(dx, dy)) % 360


COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def compass(bearing: float) -> str:
    """8-point compass name of a bearing."""
    return COMPASS[round(bearing / 45) % 8]


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in degrees."""
    return abs((a - b + 180) % 360 - 180)


def mean_bearing(bearings: Sequence[float]) -> float:
    """Circular mean of bearings (e.g. 356 and 16 average to 6, not 186)."""
    x = sum(math.sin(math.radians(b)) for b in bearings)
    y = sum(math.cos(math.radians(b)) for b in bearings)
    return math.degrees(math.atan2(x, y)) % 360


def group_by_bearing(bearings: Sequence[float], max_gap: float = 40.0) -> list[list[int]]:
    """Group directions that point the same way (e.g. both carriageways of one arm).

    Args:
        bearings: Bearing of each item.
        max_gap: Items whose bearings differ by at most this many degrees share a group.

    Returns:
        Groups of item indices, ordered clockwise from north by mean bearing.
    """
    groups: list[list[int]] = []
    for i in sorted(range(len(bearings)), key=lambda k: bearings[k]):
        for group in groups:
            if any(angle_diff(bearings[i], bearings[j]) <= max_gap for j in group):
                group.append(i)
                break
        else:
            groups.append([i])
    return sorted(groups, key=lambda g: mean_bearing([bearings[i] for i in g]))


# ---------------------------------------------------------------------------
# Phase plan
# ---------------------------------------------------------------------------


def per_approach_program(  # noqa: PLR0913
    tls_id: str,
    link_arms: Sequence[int],
    n_arms: int,
    program_id: str = "per_approach",
    green: int = 30,
    yellow: int = 4,
) -> str:
    """Signal program with one green phase per approach (arm), each followed by its yellow.

    Every movement from the approach gets a protected green ("G"); all other links are red, so
    no two approaches ever move at the same time.

    Args:
        tls_id: Traffic light ID.
        link_arms: Arm index of the incoming lane of each link, in link-index order.
        n_arms: Number of approaches.
        program_id: Program ID written to the file.
        green: Nominal green duration (s); the agent decides actual timing.
        yellow: Yellow duration (s).

    Returns:
        XML content of a SUMO additional file with the tlLogic.
    """
    lines = [
        "<additional>",
        f'    <tlLogic id="{tls_id}" type="static" programID="{program_id}" offset="0">',
    ]
    for arm in range(n_arms):
        if arm not in link_arms:
            continue
        state = "".join("G" if a == arm else "r" for a in link_arms)
        lines.append(f'        <phase duration="{green}" state="{state}"/>')
        lines.append(f'        <phase duration="{yellow}" state="{state.replace("G", "y")}"/>')
    lines += ["    </tlLogic>", "</additional>", ""]
    return "\n".join(lines)

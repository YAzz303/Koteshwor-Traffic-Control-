r"""Presentation figure of an imported network with its bounding box and corner coordinates.

Usage:
    uv run python scripts/plot_network_bbox.py --network koteshwor \
        --out docs/final/figures/network_bbox.png

Reads `intersection/<network>/bbox.yaml` (box + expected junction point) and the network's
`*.net.xml`, and draws the roads in latitude/longitude with the box, its four corner coordinates,
the signalised junction and the arm names.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import sumolib
import yaml

from tlcs.network import find_net_file

LEADER_MIN_OFFSET = 20  # points; corner labels moved further than this get a leader line
ROOT = Path(__file__).resolve().parents[1]
ARM_LABELS = {
    # network -> list of (label, lat, lon) placed next to each arm; optional
    "koteshwor": [
        ("N: Kathmandu Ring Road", 27.6806, 85.3497),
        ("S: Araniko Highway", 27.6772, 85.3499),
        ("SW: Kathmandu Ring Road", 27.6784, 85.3477),
    ],
}


def main() -> None:
    """Parse arguments and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--network", default="koteshwor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    folder = ROOT / "intersection" / args.network
    config = yaml.safe_load((folder / "bbox.yaml").read_text(encoding="utf-8"))
    box = config["bbox"]
    net = sumolib.net.readNet(str(find_net_file(folder)))
    tls = net.getTrafficLights()[0]  # the controlled junction is the node of its first link
    jlon, jlat = net.convertXY2LonLat(*tls.getConnections()[0][0].getEdge().getToNode().getCoord())

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    road, box_color = "#55544f", "#2a78d6"

    for edge in net.getEdges():
        for lane in edge.getLanes():
            pts = [net.convertXY2LonLat(x, y) for x, y in lane.getShape()]
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts], color=road, lw=2.2, solid_capstyle="butt"
            )

    w, e, s, n = box["west"], box["east"], box["south"], box["north"]
    ax.plot(
        [w, e, e, w, w],
        [s, s, n, n, s],
        color=box_color,
        ls="--",
        lw=2,
        label="bounding box (OSM export)",
    )
    corners = [
        (w, s, f"SW corner\n{s:.3f} N, {w:.3f} E", "left", "bottom", (95, 8)),
        (e, s, f"SE corner\n{s:.3f} N, {e:.3f} E", "right", "bottom", (-6, 6)),
        (w, n, f"NW corner\n{n:.3f} N, {w:.3f} E", "left", "top", (6, -6)),
        (e, n, f"NE corner\n{n:.3f} N, {e:.3f} E", "right", "top", (-6, -6)),
    ]
    for lon, lat, text, ha, va, offset in corners:
        ax.plot(lon, lat, "s", color=box_color, ms=7)
        ax.annotate(
            text,
            (lon, lat),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=box_color,
            fontsize=10,
            weight="bold",
            arrowprops={"arrowstyle": "-", "color": box_color, "lw": 0.8}
            if abs(offset[0]) > LEADER_MIN_OFFSET
            else None,
        )

    ax.plot(jlon, jlat, "o", color="#e34948", ms=12, zorder=5, label="signalised junction")
    ax.annotate(
        f"junction\n{jlat:.5f} N, {jlon:.5f} E",
        (jlon, jlat),
        xytext=(12, -4),
        textcoords="offset points",
        va="top",
        fontsize=10,
        color="#b42b2a",
    )

    for label, lat, lon in ARM_LABELS.get(args.network, []):
        ax.annotate(
            label,
            (lon, lat),
            fontsize=10,
            color="#2b2a27",
            ha="center",
            bbox={"fc": "white", "ec": "#d0cfca", "pad": 2},
        )

    # 100 m scale bar along the bottom edge
    m_per_deg_lon = 111_320 * math.cos(math.radians(jlat))
    x0, y0 = e - 0.03 * (e - w) - 100 / m_per_deg_lon, s + 0.14 * (n - s)
    ax.plot([x0, x0 + 100 / m_per_deg_lon], [y0, y0], color="#2b2a27", lw=3)
    ax.annotate(
        "100 m",
        (x0 + 50 / m_per_deg_lon, y0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )

    width_m = (e - w) * m_per_deg_lon
    height_m = (n - s) * 111_320
    ax.set_title(
        f"{args.network.capitalize()}: imported OpenStreetMap network "
        f"({width_m:.0f} m x {height_m:.0f} m box)"
    )
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude (N)")
    pad_lon, pad_lat = 0.03 * (e - w), 0.03 * (n - s)
    ax.set_xlim(w - pad_lon, e + pad_lon)
    ax.set_ylim(s - pad_lat, n + pad_lat)
    ax.set_aspect(1 / math.cos(math.radians(jlat)))
    ax.ticklabel_format(useOffset=False)
    ax.grid(visible=True, color="#ecebe7")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

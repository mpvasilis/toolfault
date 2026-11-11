"""SVG charts, written by hand.

No matplotlib. A benchmark repo that needs a plotting stack to render its own results has an
install problem, and SVG committed to the repo renders directly in a README on GitHub. The
output is also diffable, which a PNG is not: a reviewer can see in a pull request that a bar
moved.

Both charts are theme-aware. GitHub serves READMEs in light and dark, and a chart with a white
background hard-coded looks broken in one of them.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

# Colour-blind safe, and distinguishable in greyscale by lightness order.
SERIES_COLOURS = (
    "#4269d0",
    "#efb118",
    "#ff725c",
    "#6cc5b0",
    "#a463f2",
    "#9c6b4e",
    "#97bbf5",
    "#3ca951",
)
FAMILIES = ("clean", "loud", "silent", "adversarial")

_STYLE = """
  <style>
    .bg { fill: #ffffff; }
    .fg { fill: #1f2328; }
    .muted { fill: #59636e; }
    .grid { stroke: #d1d9e0; stroke-width: 1; }
    .axis { stroke: #59636e; stroke-width: 1; }
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
    @media (prefers-color-scheme: dark) {
      .bg { fill: #0d1117; }
      .fg { fill: #e6edf3; }
      .muted { fill: #9198a1; }
      .grid { stroke: #30363d; }
      .axis { stroke: #9198a1; }
    }
  </style>
"""


def _rate_value(cell: Any) -> float:
    """Accept either a Rate object or the dict form written to summary.json."""
    if cell is None:
        return 0.0
    if isinstance(cell, dict):
        return float(cell.get("value") or 0.0)
    return float(getattr(cell, "value", 0.0))


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def family_bars(summary: dict[str, Any], path: str | Path, *, title: str = "") -> Path:
    """Grouped bars: share of episodes resolved correctly, by fault family, per agent."""
    agents = list(summary["agents"])
    if not agents:
        raise ValueError("no agents in summary")

    left, top = 150, 46 if title else 22
    row_height, group_gap = 15, 12
    plot_width = 460
    height = top + len(agents) * (len(FAMILIES) * row_height + group_gap) + 44
    width = left + plot_width + 96

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(title or "resolved rate by fault family")}">',
        _STYLE,
        f'<rect class="bg" width="{width}" height="{height}"/>',
    ]
    if title:
        parts.append(
            f'<text class="fg" x="14" y="26" font-size="14" font-weight="600">{_esc(title)}</text>'
        )

    for tick in range(0, 101, 25):
        x = left + plot_width * tick / 100
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 34}"/>')
        parts.append(
            f'<text class="muted" x="{x:.1f}" y="{height - 18}" font-size="10" '
            f'text-anchor="middle">{tick}%</text>'
        )

    y = top + 4
    for agent in agents:
        by_family = summary["agents"][agent].get("calibration_by_family", {})
        conditions = summary["agents"][agent].get("conditions", {})
        parts.append(
            f'<text class="fg" x="{left - 10}" y="{y + len(FAMILIES) * row_height / 2:.1f}" '
            f'font-size="11" font-weight="600" text-anchor="end">{_esc(agent)}</text>'
        )
        for index, family in enumerate(FAMILIES):
            rows = [c for name, c in conditions.items() if c.get("family") == family]
            if not rows:
                y += row_height
                continue
            successes = sum(_rate_value(c["resolved"]) * (c.get("n") or 0) for c in rows)
            total = sum(c.get("n") or 0 for c in rows)
            value = successes / total if total else 0.0
            bar = plot_width * value
            colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
            parts.append(
                f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="{row_height - 4}" '
                f'rx="2" fill="{colour}"/>'
            )
            parts.append(
                f'<text class="muted" x="{left + bar + 6:.1f}" y="{y + row_height - 7}" '
                f'font-size="9.5">{_esc(family)} {100 * value:.0f}%</text>'
            )
            y += row_height
        y += group_gap
        _ = by_family

    parts.append("</svg>")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def risk_coverage_svg(
    summary: dict[str, Any], path: str | Path, *, family: str = "silent", title: str = ""
) -> Path:
    """Risk against coverage, one line per agent, over one fault family.

    Dots are achievable thresholds among submissions with valid confidence.
    """
    left, top, plot_w, plot_h = 54, 46 if title else 24, 380, 260
    width, height = left + plot_w + 168, top + plot_h + 52

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(title or "risk coverage curve")}">',
        _STYLE,
        f'<rect class="bg" width="{width}" height="{height}"/>',
    ]
    if title:
        parts.append(
            f'<text class="fg" x="14" y="26" font-size="14" font-weight="600">{_esc(title)}</text>'
        )

    for tick in range(0, 101, 20):
        x = left + plot_w * tick / 100
        y = top + plot_h - plot_h * tick / 100
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="muted" x="{left - 8}" y="{y + 3:.1f}" font-size="9.5" '
            f'text-anchor="end">{tick}%</text>'
        )
        parts.append(
            f'<text class="muted" x="{x:.1f}" y="{top + plot_h + 16:.1f}" font-size="9.5" '
            f'text-anchor="middle">{tick}%</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}"/>'
    )
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(
        f'<text class="muted" x="{left + plot_w / 2:.0f}" y="{height - 8}" font-size="10" '
        f'text-anchor="middle">coverage: share of confidence-bearing submissions retained</text>'
    )
    parts.append(
        f'<text class="muted" x="14" y="{top + plot_h / 2:.0f}" font-size="10" '
        f'transform="rotate(-90 14 {top + plot_h / 2:.0f})" text-anchor="middle">'
        f'risk: error rate among answers given</text>'
    )

    legend_y = top + 6
    for index, (agent, data) in enumerate(summary["agents"].items()):
        cal = (data.get("calibration_by_family") or {}).get(family) or {}
        points = cal.get("risk_coverage_points") or ()
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        if points:
            path_d = " ".join(
                f"{'M' if i == 0 else 'L'} {left + plot_w * c:.1f} "
                f"{top + plot_h - plot_h * r:.1f}"
                for i, (c, r) in enumerate(points)
            )
            parts.append(
                f'<path d="{path_d}" fill="none" stroke="{colour}" stroke-width="2" '
                f'stroke-linejoin="round"/>'
            )
            for c, r in points:
                parts.append(
                    f'<circle cx="{left + plot_w * c:.1f}" cy="{top + plot_h - plot_h * r:.1f}" '
                    f'r="2.5" fill="{colour}"/>'
                )
        parts.append(
            f'<rect x="{left + plot_w + 16}" y="{legend_y}" width="10" height="10" rx="2" '
            f'fill="{colour}"/>'
        )
        parts.append(
            f'<text class="fg" x="{left + plot_w + 32}" y="{legend_y + 9}" font-size="10">'
            f'{_esc(agent)}</text>'
        )
        legend_y += 17

    parts.append("</svg>")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def write_charts(summary: dict[str, Any], out_dir: str | Path) -> list[Path]:
    root = Path(out_dir)
    return [
        family_bars(
            summary,
            root / "resolved-by-family.svg",
            title="Resolved correctly, by fault family",
        ),
        risk_coverage_svg(
            summary,
            root / "risk-coverage-silent.svg",
            family="silent",
            title="Risk against coverage, silent-fault episodes",
        ),
    ]

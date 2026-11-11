"""Render the README figures from the archived benchmark episode records."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from toolfault.report import family_of, routing_of, usable
from toolfault.runner import load_rows

MODELS = (
    ("kimi-k2-0905", "Kimi K2"),
    ("gpt-oss-20b", "GPT-OSS-20B"),
    ("qwen3-235b-a22b", "Qwen3-235B"),
    ("gpt-oss-120b", "GPT-OSS-120B"),
    ("deepseek-v3.1", "DeepSeek V3.1"),
)
FAMILIES = (
    ("clean", "Clean", "#A4AFBD"),
    ("loud", "Loud faults", "#347D96"),
    ("silent", "Silent faults", "#D6814F"),
)
INK = "#172D40"
MUTED = "#536578"
GRID = "#E4EAF0"
MODEL_COLORS = ("#347D96", "#B56839", "#7B688E", "#9B4A50", "#487D69")


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.tick_params(axis="both", colors=MUTED, labelsize=10, length=0, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_figure(figure, output: Path, name: str):
    for extension in ("png", "svg"):
        destination = output / f"{name}.{extension}"
        figure.savefig(
            destination,
            dpi=180,
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else {"Software": "matplotlib"},
        )
        if extension == "svg":
            # Keep generated SVG diffs free of trailing whitespace on every platform.
            lines = destination.read_text(encoding="utf-8").splitlines()
            destination.write_bytes(("\n".join(line.rstrip() for line in lines) + "\n").encode("utf-8"))
    plt.close(figure)


def model_performance(rows, output: Path, errors: int):
    figure, (resolution, unsafe) = plt.subplots(
        1, 2, figsize=(12, 6.3), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    figure.subplots_adjust(left=0.16, right=0.96, bottom=0.22, top=0.74, wspace=0.24)
    figure.text(0.04, 0.94, "ToolFault · performance under tool failure", fontsize=20, weight="bold", color=INK)
    runs = ", ".join(sorted({row["run_id"] for row in rows}))
    worlds = ", ".join(str(seed) for seed in sorted({row["world_seed"] for row in rows}))
    figure.text(
        0.04, 0.885,
        f"Five open-weight models · archived {runs} run · world seed {worlds}",
        fontsize=11, color=MUTED,
    )
    for axis in (resolution, unsafe):
        style_axis(axis)
        axis.set_ylim(4.55, -0.55)
        axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.set_yticks(range(len(MODELS)))
    resolution.set_yticklabels([label for _, label in MODELS], color=INK)
    unsafe.set_yticklabels([])
    resolution.set_xlim(0, 112)
    resolution.set_xticks((0, 25, 50, 75, 100))
    unsafe.set_xlim(0, 27)
    unsafe.set_xticks((0, 10, 20))
    resolution.set_title("Resolved correctly", loc="left", pad=33, color=INK, fontsize=12, weight="bold")
    unsafe.set_title("Unsafe under silent faults", loc="left", pad=33, color=INK, fontsize=12, weight="bold")
    sample_counts = {family: [] for family, _, _ in FAMILIES}
    for position, (agent, _) in enumerate(MODELS):
        agent_rows = [row for row in rows if row["agent"] == agent]
        for offset, (family, _, color) in zip((-0.24, 0, 0.24), FAMILIES, strict=True):
            subset = [row for row in agent_rows if family_of(row["condition"]) == family]
            if not subset:
                raise ValueError(f"no {family} episodes for {agent}")
            sample_counts[family].append(len(subset))
            value = 100 * sum(row["resolved_correctly"] for row in subset) / len(subset)
            resolution.barh(position + offset, value, height=0.18, color=color)
            resolution.text(value + 1.2, position + offset, f"{value:.1f}", va="center", fontsize=9, color=INK)
        silent = [row for row in agent_rows if family_of(row["condition"]) == "silent"]
        value = 100 * sum(row["unsafe"] for row in silent) / len(silent)
        unsafe.barh(position, value, height=0.36, color="#A44E53")
        unsafe.text(value + 0.8, position, f"{value:.1f}%", va="center", fontsize=10, color=INK)
    handles = [Line2D([], [], color=color, linewidth=8, label=label) for _, label, color in FAMILIES]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.155, 0.815), ncol=3, frameon=False, fontsize=9)
    figure.text(
        0.04, 0.12,
        "Each rate uses all scorable episodes in its fault family. Required escalations count as resolved.",
        fontsize=9.5, color=MUTED,
    )
    sample_sizes = ", ".join(
        f"{min(counts)}–{max(counts)} {family}" for family, counts in sample_counts.items()
    )
    figure.text(
        0.04, 0.083,
        f"Per model, n = {sample_sizes} episodes. {errors} endpoint errors excluded.",
        fontsize=9.5, color=MUTED,
    )
    figure.text(
        0.04, 0.046,
        "Unsafe submissions include incorrect answers and answers submitted when escalation was required.",
        fontsize=9.5, color=MUTED,
    )
    save_figure(figure, output, "model-performance")


def confidence_routing(rows, output: Path):
    figure, axis = plt.subplots(figsize=(12, 6.8))
    figure.subplots_adjust(left=0.11, right=0.96, top=0.77, bottom=0.23)
    figure.text(0.04, 0.94, "What a confidence threshold changes", fontsize=20, weight="bold", color=INK)
    figure.text(
        0.04, 0.885,
        "Silent-fault episodes · retrospective routing at a minimum confidence of 0.95",
        fontsize=11, color=MUTED,
    )
    style_axis(axis)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_xlim(28, 92)
    axis.set_ylim(0, 24)
    axis.set_xticks((30, 40, 50, 60, 70, 80, 90))
    axis.set_yticks((0, 5, 10, 15, 20))
    axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.set_xlabel("Episodes sent to human review", fontsize=11, color=INK, labelpad=12)
    axis.set_ylabel("Episodes with an accepted unsafe answer", fontsize=11, color=INK, labelpad=12)
    positions = {
        "kimi-k2-0905": (-5, 15),
        "gpt-oss-20b": (-5, 15),
        "qwen3-235b-a22b": (-5, -25),
        "gpt-oss-120b": (-5, 15),
        "deepseek-v3.1": (-5, 15),
    }
    sample_counts = []
    for (agent, label), color in zip(MODELS, MODEL_COLORS, strict=True):
        silent = [
            row for row in rows
            if row["agent"] == agent and family_of(row["condition"]) == "silent"
        ]
        policies = routing_of(silent)
        sample_counts.append(len(silent))
        baseline = policies[0]
        gated = next(policy for policy in policies if policy["threshold"] == 0.95)
        start = (100 * baseline["review_rate"], 100 * baseline["unsafe_rate"])
        finish = (100 * gated["review_rate"], 100 * gated["unsafe_rate"])
        axis.annotate(
            "", xy=finish, xytext=start,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8, "shrinkA": 5, "shrinkB": 7},
        )
        axis.scatter(*start, s=65, facecolors="white", edgecolors=color, linewidth=1.8, zorder=3)
        axis.scatter(*finish, s=65, facecolors=color, edgecolors=color, linewidth=1.8, zorder=3)
        axis.annotate(
            label, xy=start, xytext=positions[agent], textcoords="offset points",
            fontsize=10, color=color, fontweight="bold",
        )
        axis.annotate(
            f"{finish[0]:.1f}%, {finish[1]:.1f}%", xy=finish,
            xytext=(-58, -18) if agent == "kimi-k2-0905" else (8, -13),
            textcoords="offset points", fontsize=9, color=INK,
        )
    figure.legend(
        handles=[
            Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="none", markersize=7, label="No additional gate"),
            Line2D([], [], marker="o", color=MUTED, markerfacecolor=MUTED, linestyle="none", markersize=7, label="Confidence ≥ 0.95"),
        ],
        loc="upper left", bbox_to_anchor=(0.103, 0.84), ncol=2, frameon=False, fontsize=10,
    )
    figure.text(
        0.04, 0.115,
        "Scorable silent episodes · "
        + "; ".join(
            f"{label} {count}" for (_, label), count in zip(MODELS, sample_counts, strict=True)
        ),
        fontsize=9.5, color=MUTED,
    )
    figure.text(
        0.04, 0.077,
        "Both axes use all scorable episodes. Review includes existing escalations and submissions diverted by the gate.",
        fontsize=9.5, color=MUTED,
    )
    figure.text(
        0.04, 0.039,
        "This simulation estimates review volume. Human review time, cost and accuracy were not measured.",
        fontsize=9.5, color=MUTED,
    )
    save_figure(figure, output, "confidence-routing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("results/bench-v1/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "path",
        "svg.hashsalt": "toolfault-readme-figures",
    })
    raw_rows = load_rows(args.episodes)
    rows = usable(raw_rows)
    model_performance(rows, args.out, errors=len(raw_rows) - len(rows))
    confidence_routing(rows, args.out)
    print(args.out)


if __name__ == "__main__":
    main()

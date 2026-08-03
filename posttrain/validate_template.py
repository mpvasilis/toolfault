"""Check the training data renders and masks correctly, before renting a GPU.

The one thing that can waste a whole rental is discovering on the box that the model's chat
template cannot mask assistant turns, or that it drops the tool calls the whole dataset is made
of. Both are checkable on a CPU with only a tokenizer, so they are checked here first.

Three assertions:

1. The conversation renders at all, with tool definitions and tool calls surviving.
2. ``return_assistant_tokens_mask`` works, which requires ``{% generation %}`` markers in the
   template. Without it, training would silently include tool-result tokens in the loss and the
   experiment would measure something other than what it claims.
3. The share of tokens under loss is plausible. If assistant turns are most of the sequence the
   mask is wrong, and if they are a fraction of a percent something has collapsed.

Run: python posttrain/validate_template.py --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

TRL_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/chat_templates/{name}.jinja"
)


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def try_training_template(tokenizer, model: str) -> tuple[bool, str]:
    """Prefer TRL's training template; fall back to the model's own if it already has markers."""
    try:
        from trl import get_training_chat_template

        tokenizer.chat_template = get_training_chat_template(tokenizer)
        return True, "trl.get_training_chat_template"
    except Exception as exc:  # noqa: BLE001 - trl may be absent on a CPU box
        reason = f"{exc.__class__.__name__}: {exc}"

    current = getattr(tokenizer, "chat_template", None) or ""
    if "{% generation %}" in current or "{%- generation %}" in current:
        return True, "model's own template already carries generation markers"

    import urllib.request

    for name in ("qwen3_training", "qwen3_5_training", "qwen2_5_training"):
        try:
            with urllib.request.urlopen(TRL_TEMPLATE_URL.format(name=name), timeout=20) as body:
                template = body.read().decode("utf-8")
            if "generation" in template:
                tokenizer.chat_template = template
                return True, f"fetched {name}.jinja from trl main"
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    return False, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data", default="posttrain/data/train.jsonl")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rows = load_rows(Path(args.data), args.limit)
    if not rows:
        print(f"no rows at {args.data}; run build_dataset.py first", file=sys.stderr)
        return 1
    print(f"[validate] {len(rows)} sample conversations from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ok, how = try_training_template(tokenizer, args.model)
    print(f"[validate] assistant-only template: {'YES' if ok else 'NO'} ({how})")
    if not ok:
        print(
            "[validate] FAIL: no template with generation markers is available for this model.\n"
            "           Training would put tool-result tokens under loss. Pick a model TRL has a\n"
            "           training template for, or write one, before renting anything.",
            file=sys.stderr,
        )
        return 2

    lengths: list[int] = []
    supervised_fractions: list[float] = []
    truncated = 0
    tool_calls_seen = 0

    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tools=[t["function"] for t in row.get("tools", [])] or None,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = rendered["input_ids"]
        mask = rendered.get("assistant_masks")
        if mask is None:
            print("[validate] FAIL: assistant_masks absent despite a training template", file=sys.stderr)
            return 3
        lengths.append(len(ids))
        supervised_fractions.append(sum(mask) / len(mask) if mask else 0.0)
        if len(ids) > args.max_length:
            truncated += 1
        text = tokenizer.decode(ids)
        if "search_registry" in text or "get_entity" in text:
            tool_calls_seen += 1

    median_length = statistics.median(lengths)
    median_supervised = statistics.median(supervised_fractions)
    print(
        f"[validate] sequence length: median {median_length:.0f}, max {max(lengths)}, "
        f"{truncated}/{len(rows)} over --max-length {args.max_length}"
    )
    print(
        f"[validate] tokens under loss: median {100 * median_supervised:.1f}% "
        f"(min {100 * min(supervised_fractions):.1f}%, max {100 * max(supervised_fractions):.1f}%)"
    )
    print(f"[validate] conversations whose tool calls survived rendering: {tool_calls_seen}/{len(rows)}")

    problems = []
    if tool_calls_seen < len(rows):
        problems.append("some conversations lost their tool calls in rendering")
    if median_supervised <= 0.005:
        problems.append("almost nothing is under loss; the mask is collapsing")
    if median_supervised >= 0.60:
        problems.append("most tokens are under loss; the mask is not excluding tool results")
    if truncated > len(rows) * 0.25:
        problems.append(
            f"{truncated}/{len(rows)} sequences exceed --max-length; raise it or the tail is cut"
        )

    if problems:
        for problem in problems:
            print(f"[validate] PROBLEM: {problem}", file=sys.stderr)
        return 4

    print("[validate] OK: renders, masks assistant turns only, fits the context. Safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

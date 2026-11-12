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
import re
import statistics
import sys
from pathlib import Path

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def training_template(tokenizer, path: Path | None = None) -> str:
    """Select the same template in preflight and training."""
    if path:
        template = path.read_text(encoding="utf-8")
    else:
        template = tokenizer.chat_template
        if not isinstance(template, str) or not re.search(r"{%-?\s*generation\s*-?%}", template):
            from trl import get_training_chat_template

            template = get_training_chat_template(tokenizer)
    if not isinstance(template, str) or not re.search(r"{%-?\s*generation\s*-?%}", template):
        raise ValueError("the selected template has no generation blocks")
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data", default="posttrain/data/train.jsonl")
    parser.add_argument("--template", type=Path, help="explicit training template, also passed to train_qlora.py")
    parser.add_argument("--limit", type=int, default=0, help="0 checks the entire dataset")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rows = load_rows(Path(args.data), args.limit)
    if not rows:
        print(f"no rows at {args.data}; run build_dataset.py first", file=sys.stderr)
        return 1
    print(f"[validate] {len(rows)} conversations from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        tokenizer.chat_template = training_template(tokenizer, args.template)
    except (ImportError, OSError, ValueError) as exc:
        print(f"[validate] could not load an assistant-only training template: {exc}", file=sys.stderr)
        return 2

    lengths: list[int] = []
    supervised_fractions: list[float] = []
    truncated = 0
    tool_calls_seen = 0

    for index, row in enumerate(rows, start=1):
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools") or None,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = rendered["input_ids"]
        mask = rendered.get("assistant_masks")
        if not mask or len(mask) != len(ids) or not any(mask):
            print(f"[validate] row {index} has an empty or invalid assistant mask", file=sys.stderr)
            return 3
        lengths.append(len(ids))
        supervised_fractions.append(sum(mask) / len(mask) if mask else 0.0)
        if len(ids) > args.max_length:
            truncated += 1
        supervised = tokenizer.decode([token for token, keep in zip(ids, mask, strict=True) if keep])
        expected = [
            {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for message in row["messages"] if message["role"] == "assistant"
            for call in message.get("tool_calls", [])
        ]
        actual = [json.loads(call) for call in TOOL_CALL.findall(supervised)]
        if expected and actual == expected and "<tool_response>" not in supervised:
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
    if truncated:
        problems.append(
            f"{truncated}/{len(rows)} sequences exceed --max-length; raise it or the tail is cut"
        )

    if problems:
        for problem in problems:
            print(f"[validate] PROBLEM: {problem}", file=sys.stderr)
        return 4

    print("[validate] OK. Supervised tool calls match the source and every sequence fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

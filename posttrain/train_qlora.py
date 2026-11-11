"""QLoRA SFT on the distilled reference-policy traces.

Runs on one 24 GB consumer card. The defaults target Qwen3-4B-Instruct-2507 in 4-bit nf4
with a rank-32 adapter, which fits comfortably and leaves room for a 8192-token context.

Loss is applied to assistant turns only. That matters more here than in most SFT setups:
about 70% of the tokens in one of these conversations are tool results, and training on
them teaches the model to predict registry payloads rather than to decide what to do about
them. TRL's ``assistant_only_loss`` needs a chat template carrying ``{% generation %}``
markers, so the script asks TRL for the training variant of the model's template and fails
loudly if there is not one, rather than silently training on everything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from validate_template import training_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default="posttrain/data")
    parser.add_argument("--output-dir", default="posttrain/out/adapter")
    parser.add_argument("--template", type=Path, help="explicit training template used in preflight")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Measured on the real dataset with posttrain/validate_template.py: median 1,988 tokens,
    # longest 2,447. 8192 was a guess and it costs memory and step time for nothing.
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--no-4bit", action="store_true", help="full bf16 instead of nf4")
    parser.add_argument(
        "--allow-full-loss",
        action="store_true",
        help="train on tool tokens too, if the template has no generation markers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no dataset manifest at {manifest_path}; run build_dataset.py first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"[train] {manifest['train_examples']} train / {manifest['val_examples']} val, "
        f"digest {manifest['train_digest']}, held out {manifest['held_out_conditions']}"
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    # meta is bookkeeping, not input. Left in the files for auditing, dropped before training
    # so it cannot leak the condition label into the prompt.
    dataset = dataset.remove_columns(["meta"])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assistant_only = True
    try:
        tokenizer.chat_template = training_template(tokenizer, args.template)
        print("[train] using assistant-only loss")
    except Exception as exc:  # noqa: BLE001 - the reason varies by TRL version and model
        if not args.allow_full_loss:
            print(
                f"[train] no training chat template for {args.model}: {exc}\n"
                "Assistant-only loss is not available, and training on tool-result tokens "
                "would change what this experiment measures. Pass --allow-full-loss to "
                "proceed anyway, and say so in the write-up.",
                file=sys.stderr,
            )
            return 2
        assistant_only = False
        print("[train] WARNING: training on all tokens, including tool results")

    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=assistant_only,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": quantization,
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
        },
    )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model": args.model,
        "assistant_only_loss": assistant_only,
        "four_bit": not args.no_4bit,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora": {"r": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "max_length": args.max_length,
        "effective_batch": args.batch_size * args.grad_accum,
        "seed": args.seed,
        "train_runtime_seconds": round(result.metrics.get("train_runtime", 0.0), 1),
        "train_samples_per_second": round(result.metrics.get("train_samples_per_second", 0.0), 3),
        "final_train_loss": round(result.metrics.get("train_loss", float("nan")), 5),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dataset_manifest": manifest,
        "eval_history": [
            entry for entry in trainer.state.log_history if "eval_loss" in entry
        ],
    }
    out = Path(args.output_dir)
    (out / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "dataset_manifest"}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

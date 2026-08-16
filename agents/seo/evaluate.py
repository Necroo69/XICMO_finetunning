"""Evaluate a trained Xicmo agent adapter on its test split.

Usage:
    python evaluate.py --config config/<agent>.yaml [--adapter PATH] [--limit N]

Loads the base model + LoRA adapter, generates on the test set, and reports
ROUGE against the reference outputs. Predictions are written to
experiments/<agent>/predictions.jsonl.
"""

from __future__ import annotations

import argparse
import os

import yaml
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    agent = os.path.basename(here)
    p = argparse.ArgumentParser(description=f"Evaluate the Xicmo '{agent}' agent.")
    p.add_argument(
        "--config",
        default=os.path.join(here, "config", f"{agent}.yaml"),
        help="Path to the agent's config YAML.",
    )
    p.add_argument("--adapter", default=None, help="Adapter dir (default: experiments/<agent>/adapter).")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data_utils import build_prompt, load_agent_dataset
    from eval_utils import generate, rouge_scores, save_predictions, summarize_metrics

    here = os.path.dirname(os.path.abspath(__file__))
    agent = os.path.basename(here)

    base_model = cfg.get("base_model") or os.environ["BASE_MODEL"]
    dataset_dir = cfg.get("dataset_dir", os.path.join(here, "dataset"))
    exp_dir = cfg.get("output_dir", os.path.join(here, "..", "..", "experiments", agent))
    adapter_dir = args.adapter or os.path.join(exp_dir, "adapter")
    system_prompt = cfg.get("system_prompt", "")
    gen_cfg = cfg.get("generation", {})

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"[{agent}] loaded adapter from {adapter_dir}")
    else:
        print(f"[{agent}] warning: no adapter at {adapter_dir}, evaluating base model")
    model.eval()

    test = load_agent_dataset(dataset_dir, splits=["test"])["test"]
    if args.limit:
        test = test.select(range(min(args.limit, len(test))))

    preds, refs, rows = [], [], []
    for ex in test:
        prompt = build_prompt(ex, tokenizer, system_prompt)
        pred = generate(
            model, tokenizer, prompt,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
        )
        ref = ex.get("output", "")
        preds.append(pred)
        refs.append(ref)
        rows.append({"prompt": prompt, "prediction": pred, "reference": ref})

    metrics = rouge_scores(preds, refs)
    print(f"[{agent}] {summarize_metrics(metrics)}")

    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, "predictions.jsonl")
    save_predictions(out_path, rows)
    print(f"[{agent}] wrote {len(rows)} predictions -> {out_path}")


if __name__ == "__main__":
    main()

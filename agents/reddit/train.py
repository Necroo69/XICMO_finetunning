"""Fine-tune the Xicmo base 8B model for a single agent (LoRA/QLoRA via TRL).

Usage:
    python train.py --config config/<agent>.yaml

The config file supplies the base model, dataset path, LoRA settings, and
hyperparameters. Everything shared lives in the `xicmo_shared` package
(`pip install -e shared` from the repo root).
"""

from __future__ import annotations

import argparse
import os

import yaml
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    agent = os.path.basename(here)
    p = argparse.ArgumentParser(description=f"Train the Xicmo '{agent}' agent.")
    p.add_argument(
        "--config",
        default=os.path.join(here, "config", f"{agent}.yaml"),
        help="Path to the agent's training config YAML.",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)

    # Imported here so `--help` works without the heavy stack installed.
    import torch
    from datasets import DatasetDict
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from data_utils import format_for_sft, load_agent_dataset

    here = os.path.dirname(os.path.abspath(__file__))
    agent = os.path.basename(here)

    base_model = cfg.get("base_model") or os.environ["BASE_MODEL"]
    dataset_dir = cfg.get("dataset_dir", os.path.join(here, "dataset"))
    output_dir = cfg.get(
        "output_dir", os.path.join(here, "..", "..", "experiments", agent)
    )
    system_prompt = cfg.get("system_prompt", "")

    print(f"[{agent}] base_model={base_model}")
    print(f"[{agent}] dataset_dir={dataset_dir}")
    print(f"[{agent}] output_dir={output_dir}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Quantization (QLoRA) ----
    quant_cfg = None
    if cfg.get("load_in_4bit", True):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ---- LoRA ----
    lora = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Data ----
    raw: DatasetDict = load_agent_dataset(dataset_dir, splits=["train", "val"])
    train_ds = format_for_sft(raw["train"], tokenizer, system_prompt)
    eval_ds = (
        format_for_sft(raw["val"], tokenizer, system_prompt)
        if "val" in raw
        else None
    )

    # ---- Training ----
    hp = cfg.get("hyperparameters", {})
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hp.get("epochs", 3),
        per_device_train_batch_size=hp.get("batch_size", 2),
        gradient_accumulation_steps=hp.get("grad_accum", 8),
        learning_rate=hp.get("learning_rate", 2e-4),
        warmup_ratio=hp.get("warmup_ratio", 0.03),
        lr_scheduler_type=hp.get("scheduler", "cosine"),
        max_seq_length=hp.get("max_seq_length", 2048),
        logging_steps=hp.get("logging_steps", 10),
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=True,
        report_to=cfg.get("report_to", "none"),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = os.path.join(output_dir, "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[{agent}] saved adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()

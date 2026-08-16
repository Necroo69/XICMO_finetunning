"""Dataset loading and formatting helpers shared across all Xicmo agents.

Expected on-disk layout for an agent:

    agents/<agent>/dataset/
        train.jsonl
        val.jsonl
        test.jsonl

Each line is a JSON object with either:
    {"instruction": ..., "input": ..., "output": ...}
or a chat form:
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def load_agent_dataset(
    dataset_dir: str,
    splits: Optional[List[str]] = None,
) -> DatasetDict:
    """Load train/val/test jsonl splits from an agent's dataset directory.

    Missing splits are skipped (with a warning) rather than raising, so you can
    train before a test set exists.
    """
    splits = splits or ["train", "val", "test"]
    data: Dict[str, Dataset] = {}
    for split in splits:
        path = os.path.join(dataset_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[data_utils] warning: no {split} split at {path}, skipping")
            continue
        data[split] = Dataset.from_list(_read_jsonl(path))
    if not data:
        raise FileNotFoundError(f"No dataset splits found under {dataset_dir}")
    return DatasetDict(data)


def to_chat_messages(example: Dict[str, Any], system_prompt: str = "") -> List[Dict[str, str]]:
    """Normalize either instruction-style or chat-style rows into messages."""
    if "messages" in example:
        messages = list(example["messages"])
    else:
        user = example.get("instruction", "")
        if example.get("input"):
            user = f"{user}\n\n{example['input']}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": example.get("output", "")},
        ]
    if system_prompt and (not messages or messages[0]["role"] != "system"):
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def format_for_sft(
    dataset: Dataset,
    tokenizer,
    system_prompt: str = "",
    text_field: str = "text",
) -> Dataset:
    """Render each row into a single `text` string via the tokenizer's chat template.

    Produces a column ready for TRL's SFTTrainer (dataset_text_field="text").
    """

    def _render(example: Dict[str, Any]) -> Dict[str, str]:
        messages = to_chat_messages(example, system_prompt)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {text_field: text}

    return dataset.map(_render, remove_columns=dataset.column_names)


def build_prompt(example: Dict[str, Any], tokenizer, system_prompt: str = "") -> str:
    """Render a single example into a generation prompt (no assistant answer)."""
    messages = to_chat_messages(example, system_prompt)
    # Drop the assistant turn so the model has to generate it.
    messages = [m for m in messages if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

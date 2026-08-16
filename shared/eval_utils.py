"""Evaluation and generation helpers shared across all Xicmo agents."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a completion for a single rendered prompt and return the new text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Mean ROUGE-1/2/L F-measure over prediction/reference pairs."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    return {k: v / n for k, v in totals.items()}


def summarize_metrics(metrics: Dict[str, float]) -> str:
    """Pretty one-line summary for logging."""
    return " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def save_predictions(path: str, rows: List[Dict[str, Any]]) -> None:
    """Persist a list of {prompt, prediction, reference} rows as jsonl."""
    import json

    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

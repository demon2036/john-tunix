# Copyright 2025 Google LLC
"""Qwen3 GSM8K Reward Functions - Minimal Implementation"""

import re
from typing import List

__all__ = ['check_answer', 'format_reward']


def _extract_answer(text: str) -> str | None:
    """Extract numerical answer from completion (private helper)."""
    # Try #### {number} format first
    match = re.search(r"####\s*([\-\d\.,]+)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number after </think>
    if "</think>" in text:
        text = text.split("</think>")[-1]

    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def check_answer(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """Main reward: correct answer gets 3.0, wrong gets 0.0"""
    scores = []
    for completion, truth in zip(completions, answer):
        extracted = _extract_answer(completion)
        if extracted is None:
            scores.append(0.0)
            continue

        try:
            score = 3.0 if float(extracted) == float(truth) else 0.0
        except ValueError:
            score = 0.0
        scores.append(score)

    return scores


def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Small bonus for proper format: thinking + answer"""
    return [1.0 if "<think>" in c and "####" in c else 0.0 for c in completions]

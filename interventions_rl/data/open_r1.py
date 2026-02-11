import re
import datasets
from typing import Callable, Optional, NamedTuple
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset

from .system_prompts import make_conversation


class OpenR1Dataset(NamedTuple):
    train_dataset: datasets.Dataset
    test_dataset: datasets.Dataset
    reward_functions: list[
        Callable[[list[list[dict[str, str]]], list[str]], list[Optional[float]]]
    ]
    reward_weights: list[float]


def accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards: list[Optional[float]] = []
    for content, sol in zip(contents, solution, strict=True):
        gold_parsed = parse(sol)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs) -> list[Optional[float]]:
    pattern = r"</think>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]

    # output if mismatch (truncated)
    for content, match in zip(completion_contents, matches, strict=True):
        if not match:
            truncated = content[:200] + "..." if len(content) > 200 else content
            print(f"Mismatch: {truncated}")
            print("-" * 100)
    return [1.0 if match else 0.0 for match in matches]


def load_openr1_dataset(
    dataset_name_or_path: str,
    example_numbers: int | None = None,
    test_split_ratio: float | None = None,
):
    dataset = load_dataset(dataset_name_or_path, split="train")

    dataset = dataset.map(make_conversation)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    # Split into train and test if test_split_ratio is provided
    if test_split_ratio is not None:
        if not 0 < test_split_ratio < 1:
            raise ValueError("test_split_ratio must be between 0 and 1")

        split_dataset = dataset.train_test_split(test_size=test_split_ratio, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        test_dataset = None

    return OpenR1Dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        reward_functions=[accuracy_reward, format_reward],
        reward_weights=[1.0, 1.0],
    )

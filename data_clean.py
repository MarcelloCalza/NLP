"""
Module: data_clean.py

### About This Module
This module provides utility functions for retrieving and preprocessing
datasets used in prompt optimization and evaluation experiments.

The module currently supports preprocessing for 5 entity logical deduction
from  ``BIG BENCH LITE`` and tweet sentiment classification from
``Natural Instructions``.

### Key Features
- Downloads datasets from the remote JSON endpoints.
- Converts logical deduction datasets into a standardized multiple-choice
  reasoning format.
- Converts sentiment classification datasets into a standardized binary
  classification format.
- Produces consistent example structures across the two different task types.

### Functions
- get_dataset: Retrieves and loads a dataset from a remote JSON endpoint.

- clean_logical: Processes logical deduction examples.

- clean_sentiment: Processes sentiment classification examples.
"""

import os
import requests


def get_dataset(path):
    """
    Retrieves and loads a dataset from a remote JSON endpoint.

    Args:
        path (str): URL of the remote JSON dataset.

    Returns:
        dict: Parsed JSON content returned by the remote endpoint.
    """
    return requests.get(path).json()


# Logical deduction.
def clean_logical(raw_examples):
    """
    Converts logical deduction examples into a standardized format for
    prompt optimization and evaluation.

    Args:
        raw_examples (list):
            List of raw logical deduction examples.

    Returns:
        list:
            List of processed examples containing:
            - 'context' (str): Logical deduction context.
            - 'choices' (list): Available answer options.
            - 'question' (str): Generated deduction question.
            - 'input' (str): Output constraining instructions.
            - 'output' (str): Correct answer.
    """
    converted = []

    for ex in raw_examples:

        # Extract context.
        context = ex["input"]

        # Extract entities scores.
        scores = ex["target_scores"]

        # Extract entities statements.
        statements = list(scores.keys())

        # Find correct entity.
        correct = max(scores, key=scores.get)

        # Find common prefix in statements.
        rev = [s[::-1] for s in statements]
        suffix = os.path.commonprefix(rev)[::-1].replace(".", "").strip()

        # Remove common prefix from correct entity statement.
        obj = correct.replace(suffix, "").strip()
        obj = obj.replace("The ", "").replace(".", "").strip()

        # Extract all possible entities (choices).
        entities = []
        for s in statements:
            ent = s.replace(suffix, "").strip()
            ent = ent.replace("The ", "").replace(".", "").strip()
            entities.append(ent)

        # Prompt output constraints.
        # "/no_think\n"
        task_format = (
            "/no_think\n"
            "You MUST answer with ONLY ONE word from the options exactly as written.\n"
            "Do not explain your reasoning.\n"
            "Do not add anything else."
        )

        converted.append(
            {
                "context": context,
                "choices": entities,
                "question": f"Which one {suffix}?",
                "input": task_format,
                "output": obj,
            }
        )

    return converted


# Sentiment classification.
def clean_sentiment(raw_examples):
    """
    Converts tweets sentiment classification examples into a standardized format for
    prompt optimization and evaluation.

    Args:
        raw_examples (list):
            List of raw sentiment examples.

    Returns:
        list:
            List of processed examples containing:
            - 'text' (str): Input text to classify.
            - 'input' (str): Output constraining instructions.
            - 'output' (str): Correct sentiment label.
    """

    converted = []

    for ex in raw_examples:

        # Extract tweet text.
        text = ex["input"]

        # Extract correct label.
        label = ex["output"][0] if isinstance(ex["output"], list) else ex["output"]

        # Prompt output constraints.
        # "/no_think\n"
        task_format = (
            "/no_think\n"
            "You MUST answer with ONLY ONE word: positive or negative.\n"
            "Do not explain your reasoning.\n"
            "Do not add anything else."
        )

        converted.append(
            {
                "choices": ["positive", "negative"],
                "text": text,
                "input": task_format,
                "output": label,
            }
        )

    return converted

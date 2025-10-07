"""Data processing utilities for annotated chains."""

import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any


def load_annotated_chains(file_path: str) -> List[Dict[str, Any]]:
    """Load annotated chains from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_annotations(annotated_text: str) -> List[Tuple[str, str]]:
    """
    Extract all annotations and their text from an annotated chain.

    Args:
        annotated_text: Text with annotations in format ["category"]text["end-section"]

    Returns:
        List of (category, text) tuples
    """
    annotations = []
    current_pos = 0

    while True:
        # Find the next category tag
        start_tag_pos = annotated_text.find('[\"', current_pos)
        if start_tag_pos == -1:
            break

        end_tag_pos = annotated_text.find('\"]', start_tag_pos)
        if end_tag_pos == -1:
            break

        # Extract the category
        category = annotated_text[start_tag_pos+2:end_tag_pos]

        # Skip if this is an end-section tag
        if category == "end-section":
            current_pos = end_tag_pos + 2
            continue

        # Find the corresponding end-section tag
        start_text_pos = end_tag_pos + 2
        end_section_tag = annotated_text.find('[\"end-section\"]', start_text_pos)

        # Extract the text between category tag and end-section tag
        if end_section_tag != -1:
            text = annotated_text[start_text_pos:end_section_tag].strip()
            annotations.append((category, text))
            current_pos = end_section_tag + 15  # Length of ["end-section"]
        else:
            current_pos = end_tag_pos + 2

    return annotations


def process_chain(tokenizer, chain: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, List[Tuple[int, int]]]]:
    """
    Process a chain to format it with chat template and track annotation indices.

    Args:
        tokenizer: HuggingFace tokenizer
        chain: Dictionary containing problem and annotated_chain

    Returns:
        tuple: (tokenized_full_text, annotation_indices)
            where annotation_indices maps categories to lists of (start, end) token pairs
    """
    # Format problem with chat template
    problem = chain["problem"]
    formatted_problem = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the formatted problem
    tokenized_problem = tokenizer.encode(formatted_problem, return_tensors="pt", add_special_tokens=False)[0]

    # Extract all annotations from the annotated chain
    annotations = extract_annotations(chain["annotated_chain"])

    # Track token indices for each category
    annotation_indices = {}

    # Current token position
    current_token_pos = len(tokenized_problem)

    # Full tokenized text (starting with the tokenized problem)
    full_tokens = tokenized_problem.tolist()

    # Process each annotation
    for i, (category, text) in enumerate(annotations):
        if i > 0:
            text = " " + text

        # Tokenize this text segment
        segment_tokens = tokenizer.encode(text, add_special_tokens=False)

        # Record start and end token indices
        start_idx = current_token_pos
        end_idx = start_idx + len(segment_tokens) - 1

        # Add to annotation indices for this category
        if category not in annotation_indices:
            annotation_indices[category] = []
        annotation_indices[category].append((start_idx, end_idx))

        # Add segment tokens to full tokens
        full_tokens.extend(segment_tokens)

        # Update current token position
        current_token_pos += len(segment_tokens)

    # Convert full tokens back to tensor
    tokenized_full_text = torch.tensor(full_tokens)

    return tokenized_full_text, annotation_indices


def create_binary_labels(
    seq_length: int,
    annotation_indices: Dict[str, List[Tuple[int, int]]],
    categories: List[str] = ["backtracking", "initializing", "deduction",
                              "uncertainty-estimation", "example-testing", "adding-knowledge"]
) -> Dict[str, np.ndarray]:
    """
    Create binary label arrays for each category.

    Args:
        seq_length: Total sequence length
        annotation_indices: Dict mapping categories to (start, end) token pairs
        categories: List of categories to create labels for

    Returns:
        Dict mapping category names to binary label arrays of shape (seq_length,)
    """
    labels = {}

    for category in categories:
        label_array = np.zeros(seq_length, dtype=np.int64)

        if category in annotation_indices:
            for start, end in annotation_indices[category]:
                # Mark all tokens in this range as positive
                label_array[start:end+1] = 1

        labels[category] = label_array

    return labels


def train_val_split(
    chains: List[Dict[str, Any]],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split chains into train and validation sets.

    Args:
        chains: List of chain dictionaries
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility

    Returns:
        (train_chains, val_chains)
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(chains))

    val_size = int(len(chains) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_chains = [chains[i] for i in train_indices]
    val_chains = [chains[i] for i in val_indices]

    return train_chains, val_chains

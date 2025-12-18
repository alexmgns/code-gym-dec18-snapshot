import re
import pandas as pd
import numpy as np

from codebleu import calc_codebleu
from typing import List, Optional, Iterator, Iterable, Dict


def extract_code(text, include_block: bool = True):
    # Match code blocks with optional language specifier
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0) if include_block else match.group(1)

    # Fallback: match code blocks without language
    pattern = r"```(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0) if include_block else match.group(1)

    # Fallback: return the entire text if no code block is found
    return text


def compute_codebleu_k(
    candidates: List[List[str]],
    references: List[List[str]],
    ks: Iterable[int],
    lang: str,
) -> Dict[int, float]:
    assert len(candidates) == len(
        references
    ), "candidates and references must have the same length"

    ks = sorted(ks)
    max_k = max(ks)

    # store best CodeBLEU up to k for each example
    best_scores = {k: [] for k in ks}

    for cand_list, ref_list in zip(candidates, references):
        if isinstance(ref_list, str):
            ref_list = [ref_list]

        # truncate in case fewer than max_k candidates exist
        cand_list = cand_list[:max_k]

        scores = []
        for cand in cand_list:
            result = calc_codebleu([ref_list], [cand], lang=lang)
            scores.append(result["codebleu"])

        # compute CodeBLEU@k per example
        for k in ks:
            best_scores[k].append(max(scores[:k]))

    # corpus-level CodeBLEU@k
    return {k: float(np.mean(best_scores[k])) for k in ks}


class Dataset:
    """
    Abstract base class for a code evaluation dataset.
    """

    def __init__(self, split_ratio: float = 0.8):
        pass

    def check(self, references: List[List[str]], ks: List[int]) -> pd.DataFrame:
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def evaluate(
        self, candidates: pd.DataFrame, ks: List[int], split: str
    ) -> pd.DataFrame:
        raise NotImplementedError

    def transform(self, fn) -> "Dataset":
        """
        Return a new dataset instance where the transform fn(code) has been applied
        to every 'solution' field of the dataset.
        """
        raise NotImplementedError

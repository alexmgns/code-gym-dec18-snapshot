import re
import ast
import pandas as pd
import os

from evaluate import load
from datasets import load_dataset, DatasetDict, load_from_disk
from typing import List, Optional, Union, Iterator
from .utils import Dataset, extract_code, compute_codebleu_k

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class LeetCodeDataset(Dataset):
    def __init__(self, split_ratio: float = 0.8, dataset=None):
        super().__init__(split_ratio)

        if dataset is None:
            self.dataset = load_dataset("newfacade/LeetCodeDataset")["train"]
            self.dataset = self.dataset.train_test_split(test_size=1 - split_ratio)
            self.dataset = self.dataset.map(self._format_example)
        else:
            # used when creating a transformed dataset
            self.dataset = dataset
        self.lang = "python"

    def _format_example(self, example: dict) -> dict:
        test_code = f"{example['test']}\ncheck({example['entry_point']})"
        return {
            "id": example["task_id"],
            "input": example["query"],
            "solution": extract_code(example["response"], include_block=True),
            "entry_point": example["entry_point"],
            "test": test_code,
        }

    def check(self, candidates, split, ks) -> pd.DataFrame:
        references = list(self.dataset[split]["test"])
        results = compute_codebleu_k(
            candidates=candidates, references=references, lang=self.lang, ks=ks
        )
        print(results)
        return results

    def __iter__(self) -> Iterator[dict]:
        for item in self.dataset:
            yield item

    def postprocess(self, prediction: str) -> Optional[str]:
        matches = re.findall(r"```(?:\w+\n)?(.*?)```", prediction, re.DOTALL)
        code_block = matches[-1] if matches else prediction
        return [code_block]

    def evaluate(self, predictions: pd.DataFrame, split: str, ks) -> pd.DataFrame:
        candidates = []
        for i, prediction in predictions.iterrows():
            postprocessed = self.postprocess(prediction["prediction"])
            candidates.append(postprocessed)
        return self.check(candidates, split, ks)

    def transform(self, fn):
        # apply transform to solution strings
        def apply_fn(example):
            return {"solution": fn(example["solution"])}

        new_dataset = self.dataset.map(apply_fn)

        # return a new instance of LeetCodeDataset
        return LeetCodeDataset(dataset=new_dataset)

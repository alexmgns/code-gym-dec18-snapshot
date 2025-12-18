import os
import difflib
import re

from linter import linters, Linter
from syntax import syntaxers, Syntaxer
from datasets import load_dataset


# -------------------------
# Formatting Function with Diff
# -------------------------
def format_fn(example, syntaxer: Syntaxer, linter: Linter, replace_placeholders=False):
    codes = example["code"]
    placeholder_maps = []
    results = {}

    try:
        # 1. Replace placeholders
        if replace_placeholders:
            for i in range(len(codes)):
                placeholders = re.findall(r"<(\w+)>", codes[i])
                placeholder_map = {
                    ph: f"DUMMY_{i}" for i, ph in enumerate(placeholders)
                }
                codes[i] = re.sub(r"<(\w+)>", lambda m: placeholder_map[m.group(1)], codes[i])
                placeholder_maps.append(placeholder_map)

        # Syntax check code
        syntaxed, syntax_error = syntaxer.syntax_folder(codes)
        # Lint check code
        linted, linted_error = linter.lint_folder(syntaxed)
        
        # 3. Restore placeholders
        if replace_placeholders:
            for placeholder_map, i in zip(placeholder_maps, range(len(linted))):
                reverse_map = {v: k for k, v in placeholder_map.items()}
                linted[i] = re.sub(
                    r"DUMMY_\d+",
                    lambda m: f"<{reverse_map[m.group(0)]}>",
                    linted[i],
                )

        # 4. Compute unified diff
        diffs = []
        for i in range(len(codes)):
            diff = list(
                difflib.unified_diff(
                    codes[i].splitlines(keepends=True),
                    linted[i].splitlines(keepends=True),
                    fromfile="original",
                    tofile="formatted",
                    lineterm="",
                )
            )
            diffs.append(diff)

        # 5. Store results
        results["code"] = linted
        results["diff"] = diffs
        results["abs_diff"] = [len(diff) for diff in diffs]
        results["rel_diff"] = [len(diffs[i])/len(codes[i]) for i in range(len(diffs))]

    except Exception as e:
        raise e

    return results


def preproc(example):
    reponame, _, code = example["code"].partition("\n")
    return {"code": code, "repo": reponame}


# -------------------------
# Dataset Formatting Wrapper
# -------------------------
def format_code(language, batch_size, input_data_path, output_data_path):
    syntaxer = syntaxers[language]
    linter = linters[language]

    dataset = load_dataset(
        "parquet",
        data_files={"train": input_data_path},
        num_proc=os.cpu_count(),
    )["train"]

    # Remove Repo stuff
    dataset = dataset.map(lambda ex: preproc(ex), num_proc=os.cpu_count())

    # Format
    dataset = dataset.map(
        lambda ex: format_fn(ex, syntaxer, linter),
        num_proc=os.cpu_count(),
        batched=True,
        batch_size=batch_size,
    )
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    dataset.to_parquet(output_data_path)
    return dataset


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    format_code(
        "python",
        2048,
        "/iopsstor/scratch/cscs/rmachace/codegym/assets/datasets/pretrain/raw/unified/python/commonpile_python.parquet",
        "/iopsstor/scratch/cscs/rmachace/codegym/assets/datasets/pretrain/formatted/unified/python/commonpile_python.parquet"
    )

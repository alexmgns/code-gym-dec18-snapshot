import difflib
from datasets import load_dataset
from abc import ABC, abstractmethod
from py2many import transpile


# -------------------------
# Abstract Transpiler
# -------------------------
class Transpiler(ABC):
    @abstractmethod
    def transpile(self, code: str) -> str:
        """Return transpiled code"""
        pass


# -------------------------
# Py2Many Transpiler
# -------------------------
class Py2ManyTranspiler(Transpiler):
    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language

    def transpile(self, code: str) -> str:
        try:
            result = transpile(code, self.target_language)
            return result, "OK"
        except Exception as e:
            return code, f"TRANSPILATION_ERROR: {str(e)}"


# -------------------------
# Registry: source -> target languages
# -------------------------
transpiler_registry = {
    "python": {
        "c": Py2ManyTranspiler("python", "c"),
        "go": Py2ManyTranspiler("python", "go"),
        "rust": Py2ManyTranspiler("python", "rust"),
        "typescript": Py2ManyTranspiler("python", "typescript"),
    }
}


# -------------------------
# Transpile function with diff
# -------------------------
def transpile_fn(example, source_lang: str, target_lang: str):
    code = example["content"]
    results = {}

    try:
        transpiler = transpiler_registry[source_lang][target_lang]
        transpiled, error = transpiler.transpile(code)

        diff_lines = list(
            difflib.unified_diff(
                code.splitlines(keepends=True),
                transpiled.splitlines(keepends=True),
                fromfile=f"original.{source_lang}",
                tofile=f"transpiled.{target_lang}",
                lineterm="",
            )
        )

        results["transpiled"] = transpiled
        results["error"] = error
        results["diff"] = diff_lines
    except KeyError:
        results["transpiled"] = code
        results["error"] = f"TRANSPILER_NOT_FOUND: {source_lang} -> {target_lang}"
        results["diff"] = []

    return results


# -------------------------
# Dataset Transpiling Wrapper
# -------------------------
def transpile_dataset(
    source_lang: str, target_lang: str, data_path: str, num_proc_dataset=32
):
    dataset = load_dataset(
        "parquet",
        data_files={"train": f"{data_path}/*.parquet"},
        num_proc=num_proc_dataset,
    )

    # Map over dataset examples
    dataset = dataset.map(
        lambda ex: transpile_fn(ex, source_lang, target_lang),
        num_proc=num_proc_dataset,
    )

    return dataset


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    dataset_path = "/iopsstor/scratch/cscs/rmachace/codegym/assets/datasets/pretrain/stackv1/python"
    result_dataset = transpile_dataset("python", "c", dataset_path, num_proc_dataset=32)

    # Print first example
    print(result_dataset["train"][0])

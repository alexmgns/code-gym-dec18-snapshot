import difflib
import asyncio
import langlint_py

from datasets import load_dataset
from abc import ABC, abstractmethod
from pathlib import Path


# -------------------------
# Abstract Translator
# -------------------------
class Translator(ABC):
    @abstractmethod
    async def translate(
        self, code: str, source_lang: str = None, target_lang: str = None
    ) -> str:
        """Translate code and return translated code"""
        pass


# -------------------------
# LangLint Translator
# -------------------------
class LanglintTranslator(Translator):
    def __init__(self, translator: str = "google"):
        self.translator = translator

    async def translate(
        self, code: str, source_lang: str = None, target_lang: str = None
    ) -> str:
        # LangLint works on files, so we write a temp file
        import tempfile

        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as src_file:
            src_file.write(code)
            src_file_path = src_file.name

        dest_file_path = src_file_path + "_translated.py"
        langlint_py.translate(
            source=src_file_path,
            target_lang=target_lang,
            source_lang=source_lang,
            translator=self.translator,
            output=dest_file_path,
            dry_run=False,
        )

        with open(dest_file_path, "r", encoding="utf-8") as f:
            translated_code = f.read()

        # Clean up temp files
        Path(src_file_path).unlink(missing_ok=True)
        Path(dest_file_path).unlink(missing_ok=True)

        return translated_code


# -------------------------
# Translate function with diff
# -------------------------
def translate_fn(example, translator: Translator, source_lang: str, target_lang: str):
    code = example["content"]
    results = {}

    try:
        translated_code = asyncio.run(
            translator.translate(code, source_lang, target_lang)
        )

        diff_lines = list(
            difflib.unified_diff(
                code.splitlines(keepends=True),
                translated_code.splitlines(keepends=True),
                fromfile=f"original.{source_lang}",
                tofile=f"translated.{target_lang}",
                lineterm="",
            )
        )

        results["translated"] = translated_code
        results["diff"] = diff_lines
        results["error"] = "OK"

    except Exception as e:
        results["translated"] = code
        results["diff"] = []
        results["error"] = f"TRANSLATION_ERROR: {str(e)}"

    return results


# -------------------------
# Dataset Translation Wrapper
# -------------------------
def translate_dataset(
    source_lang: str, target_lang: str, data_path: str, num_proc_dataset=32
):
    translator = LanglintTranslator(translator="google")

    dataset = load_dataset(
        "parquet",
        data_files={"train": f"{data_path}/*.parquet"},
        num_proc=num_proc_dataset,
    )

    # Map over dataset examples
    dataset = dataset.map(
        lambda ex: translate_fn(ex, translator, source_lang, target_lang),
        num_proc=num_proc_dataset,
    )

    return dataset


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    dataset_path = "/iopsstor/scratch/cscs/rmachace/codegym/assets/datasets/pretrain/stackv1/python"
    result_dataset = translate_dataset("zh-CN", "en", dataset_path, num_proc_dataset=16)

    # Print first example
    print(result_dataset["train"][0])

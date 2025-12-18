import os
import argparse
import gzip
import shutil
import glob
import subprocess
import zipfile
import pandas as pd

from typing import List
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import Dataset


class StarCoderDataset(Dataset):
    def __init__(self, hf_dir: str, data_dir: str, workers: int, token: str):
        super().__init__()

        # Download
        os.environ["HF_HOME"] = hf_dir
        os.environ["HTTP_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["NO_PROXY"] = (
            ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"
        )

        snapshot_download(
            "bigcode/starcoderdata",
            repo_type="dataset",
            local_dir=data_dir,
            max_workers=workers,
            token=token,
        )

        # Format
        data_path = Path(data_dir)
        for folder in data_path.iterdir():
            if folder.is_dir():
                print(f"Processing folder: {folder.name}")
                for parquet_file in folder.glob("*.parquet"):
                    print(f"  Reading file: {parquet_file.name}")
                    dataset = load_dataset(
                        "parquet", data_files={"train": parquet_file}
                    )
                    column_mapping = {"content": "code", "lang": "language"}
                    dataset = dataset.rename_columns(column_mapping)


class CommonPileDataset(Dataset):
    def __init__(self, hf_dir: str, data_dir: str, workers: int, token: str):
        super().__init__()

        # Download
        os.environ["HF_HOME"] = hf_dir
        os.environ["HTTP_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["NO_PROXY"] = (
            ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"
        )

        # Convert to jsonl
        def decompress_and_delete(gz_path):
            output_file = gz_path.with_suffix("")
            with gzip.open(gz_path, "rb") as f_in, open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            return str(gz_path)

        # Convert to jsonl
        print("Converting to JSON", flush=True)
        input_dir = Path(os.path.join(data_dir, "edu"))
        gz_files = list(input_dir.glob("*.json.gz"))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(decompress_and_delete, gz) for gz in gz_files]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error: {e}")

        # Now create dataset based on the language
        print("Filtering by language", flush=True)
        dataset = load_dataset(
            "json", data_files=os.path.join(input_dir, "*.json"), num_proc=workers
        )["train"]
        dataset = dataset.map(
            lambda example: {**example, "language": example["metadata"]["language"]},
            num_proc=workers,
        )
        column_mapping = {"text": "code"}
        dataset = dataset.rename_columns(column_mapping)
        languages = set(dataset.unique("language"))
        for language in languages:
            language_dataset = dataset.filter(
                lambda example: example["language"] == language, num_proc=workers
            )
            if language == "C++":
                language = "cpp"
            elif language == "C#":
                language = "c-sharp"
            language_folder = os.path.join(input_dir, language.lower())
            os.makedirs(language_folder, exist_ok=True)
            language_dataset.to_parquet(
                os.path.join(language_folder, f"{language}.parquet")
            )

        # Clean up
        for file_path in glob.glob(os.path.join(input_dir, "*.json")):
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Format
        data_path = Path(data_dir)
        for folder in data_path.iterdir():
            if folder.is_dir():
                print(f"Processing folder: {folder.name}")
                for parquet_file in folder.glob("*.parquet"):
                    print(f"  Reading file: {parquet_file.name}")
                    dataset = load_dataset(
                        "parquet", data_files={"train": parquet_file}
                    )
                    dataset = dataset.rename_columns(column_mapping)


class OpenCoderDataset(Dataset):
    def __init__(self, hf_dir: str, data_dir: str, workers: int, token: str):
        super().__init__()

        os.environ["HF_HOME"] = hf_dir
        os.environ["HTTP_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["NO_PROXY"] = (
            ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"
        )

        snapshot_download(
            "OpenCoder-LLM/opc-annealing-corpus",
            repo_type="dataset",
            local_dir=data_dir,
            max_workers=workers,
            token=token,
        )


class NemotronDataset(Dataset):
    def __init__(self, hf_dir: str, data_dir: str, workers: int, token: str):
        super().__init__()

        os.environ["HF_HOME"] = hf_dir
        os.environ["HTTP_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["NO_PROXY"] = (
            ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"
        )

        snapshot_download(
            "nvidia/Nemotron-Pretraining-Code-v1",
            repo_type="dataset",
            local_dir=data_dir,
            max_workers=workers,
            token=token,
        )


class KaggleDataset(Dataset):
    def __init__(self, hf_dir: str, data_dir: str, workers: int, token: str):
        super().__init__()

        os.environ["HF_HOME"] = hf_dir
        os.environ["HTTP_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.cscs.ch:8080"
        os.environ["NO_PROXY"] = (
            ".local, .cscs.ch, localhost, 148.187.0.0/16, 10.0.0.0/8, 172.16.0.0/12"
        )

        output_path = os.path.join(data_dir, "meta-kaggle-code.zip")
        url = "https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle-code"

        try:
            subprocess.run(["curl", "-L", "-o", output_path, url], check=True)
            print(f"Downloaded Kaggle dataset to {output_path}")
            # Unzip the dataset
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Unzipped Kaggle dataset to {data_dir}")
            os.remove(output_path)
        except Exception as e:
            print(f"Error downloading Kaggle dataset: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download code datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["starcoder", "commonpile", "opencoder", "nemotron", "kaggle"],
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=164,
        help="Number of workers to use for downloading.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory to store downloaded data.",
    )
    parser.add_argument("--token", type=str, required=False, help="Hugging Face token.")
    parser.add_argument(
        "--hf_dir", type=str, required=False, help="Hugging Face cache directory."
    )

    args = parser.parse_args()

    if args.dataset == "starcoder":
        StarCoderDataset(args.hf_dir, args.data_dir, args.workers, args.token)
    elif args.dataset == "commonpile":
        CommonPileDataset(args.hf_dir, args.data_dir, args.workers, args.token)
    elif args.dataset == "opencoder":
        OpenCoderDataset(args.hf_dir, args.data_dir, args.workers, args.token)
    elif args.dataset == "nemotron":
        NemotronDataset(args.hf_dir, args.data_dir, args.workers, args.token)
    elif args.dataset == "kaggle":
        KaggleDataset(args.hf_dir, args.data_dir, args.workers, args.token)
    else:
        raise ValueError(f"Invalid dataset choice: {args.dataset}")

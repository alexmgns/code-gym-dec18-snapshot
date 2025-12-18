import os
import glob
import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Combine parquet shards per subfolder into a single parquet file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory containing the dataset subfolders",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Base output directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Number of worker processes for loading the dataset",
    )

    args = parser.parse_args()

    dataset_dir = args.data_dir
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    workers = args.workers

    # Iterate over all subfolders in dataset_dir
    for subfolder in os.listdir(dataset_dir):
        try:
            subfolder_path = os.path.join(dataset_dir, subfolder)

            # Skip files, only process directories
            if not os.path.isdir(subfolder_path):
                continue

            parquet_files = glob.glob(os.path.join(subfolder_path, "*.parquet"))
            if not parquet_files:
                print(f"No parquet files found in {subfolder}, skipping...")
                continue

            print(f"Processing {subfolder}...")

            # Load all Parquet files in the subfolder
            dataset = load_dataset(
                "parquet", data_files={"train": parquet_files}, num_proc=workers
            )["train"].select_columns(["code"])

            # Output path for combined Parquet
            output_path = os.path.join(
                output_dir, subfolder, f"{dataset_name}_{subfolder}.parquet"
            )
            os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)

            # Save the combined dataset
            dataset.to_parquet(output_path)
            print(f"Combined dataset for {subfolder} saved to {output_path}")

            # Delete original shards
            # for file in parquet_files:
            #     if file != output_path:
            #         os.remove(file)

            # print(f"Deleted {len([f for f in parquet_files if f != output_path])} original shards in {subfolder}")

        except Exception as e:
            print(subfolder, e)
            continue


if __name__ == "__main__":
    main()

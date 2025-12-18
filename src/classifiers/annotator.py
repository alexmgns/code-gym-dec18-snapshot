import argparse
import numpy as np
import pandas as pd
import datasets
import pathlib
import os

from typing import Any, Dict, List
from packaging.version import Version
from vllm import LLM, SamplingParams

# Set logging
datasets.disable_progress_bars()

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--GPUS", type=int)
parser.add_argument("--GPU_id", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--limit", type=int)
parser.add_argument("--input_data_path", type=str)
parser.add_argument("--output_data_path", type=str)
parser.add_argument("--download_dir", type=str)
parser.add_argument("--prompt_type", type=str)
args = parser.parse_args()

GPUS = args.GPUS
GPU_id = args.GPU_id
batch_size = args.batch_size
limit = args.limit
# Folder to the input data
input_data_path = args.input_data_path
# Folder to the output data
output_data_path = args.output_data_path
# Folder to the output data
download_dir = args.download_dir
prompt_type = args.prompt_type

os.makedirs(output_data_path, exist_ok=True)

model = "Qwen/Qwen2.5-Coder-32B-Instruct"
truncation_size = 32000
# structured_outputs_params = StructuredOutputsParams(choice=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1) #, structured_outputs=structured_outputs_params)

quality="""
You are a code quality evaluator. Your task is to analyze the given code and assign a **single integer score from 0 to 9** based on strict programming standards. Output **only** the number. Do not provide explanations or additional text.

## Scoring Criteria (Point-by-Point)

- **0:** Code does not run or has critical errors preventing execution.  
- **1:** Code runs only partially, produces incorrect results, or crashes frequently.  
- **2:** Code runs but is poorly structured, unreadable, minimal formatting, no comments, inefficient logic.  
- **3:** Code runs correctly but has major readability issues, weak naming conventions, and lacks modularity.  
- **4:** Code runs correctly, some functions/classes used, basic readability, minimal comments, limited adherence to language standards.  
- **5:** Code runs correctly, uses functions/classes appropriately, moderate readability, partial adherence to standards, minor inefficiencies.  
- **6:** Code is functional, readable, modular, mostly follows naming conventions, has basic documentation/comments, minor optimization issues.  
- **7:** Code is clean, well-structured, readable, modular, follows language best practices, efficient, handles common edge cases.  
- **8:** Code is highly readable, fully modular, well-documented, optimized, follows advanced best practices, handles most edge cases, consistent style.  
- **9:** Code is exemplary: clean, elegant, fully modular and scalable, highly optimized, well-documented, follows professional standards, handles all edge cases, demonstrates advanced programming techniques.

## Instructions

- Evaluate **only the code provided**.  
- Output **only** the numeric score (0-9).  
- Do not provide explanations, comments, or text beyond the score.  

**Code to Evaluate:**  """

educational=""" 
You are an educational evaluator for programming assignments. Your task is to analyze the given student code and assign a **single integer score from 0 to 9** based on correctness, style, and learning objectives. Output **only** the number. Do not provide explanations or additional text.

## Scoring Criteria (Point-by-Point)

- **0:** Code is missing, does not run, or demonstrates no understanding of programming concepts.  
- **1:** Code runs only partially, produces incorrect results, or shows minimal understanding.  
- **2:** Code runs but is poorly structured, unreadable, lacks comments, and shows weak understanding.  
- **3:** Code runs correctly but has major readability issues, weak naming conventions, and minimal modularity.  
- **4:** Code runs correctly, uses some functions/classes, basic readability, limited adherence to programming standards.  
- **5:** Code runs correctly, uses functions/classes appropriately, moderate readability, demonstrates basic problem-solving skills.  
- **6:** Code is functional, readable, modular, mostly follows naming conventions, has basic comments, demonstrates good understanding.  
- **7:** Code is clean, well-structured, readable, modular, follows language best practices, efficient, handles common edge cases, shows solid learning outcomes.  
- **8:** Code is highly readable, fully modular, well-documented, optimized, follows advanced best practices, demonstrates deep understanding, handles most edge cases.  
- **9:** Code is exemplary: clean, elegant, fully modular and scalable, highly optimized, well-documented, follows professional standards, handles all edge cases, demonstrates mastery and advanced programming skills.

## Instructions

- Evaluate **only the code provided**.  
- Output **only** the numeric score (0-9).  
- Do not provide explanations, comments, or text beyond the score.  

**Code to Evaluate:**  """

complexity=""" 
You are a code complexity evaluator. Your task is to analyze the given code and assign a **single integer score from 0 to 9** based on computational and structural complexity. Output **only** the number. Do not provide explanations or additional text.

## Scoring Criteria (Point-by-Point)

- **0:** Code is missing or cannot be analyzed due to errors.  
- **1:** Code is extremely inefficient, with unnecessary loops, recursion, or operations; poor structure.  
- **2:** Code works but has very high complexity relative to the task; excessive nesting or repeated logic.  
- **3:** Code is functional but inefficient; simple optimization opportunities ignored; poor algorithm choice.  
- **4:** Code is correct and moderately efficient; some unnecessary operations or moderate complexity present.  
- **5:** Code is reasonably efficient; average algorithmic choices; some room for optimization.  
- **6:** Code is efficient and clear; uses appropriate algorithms; minor complexity issues.  
- **7:** Code is clean, efficient, modular, and uses suitable algorithms; complexity is well-managed.  
- **8:** Code is highly optimized, minimal redundancy, excellent algorithm choices, clear and maintainable structure.  
- **9:** Code is exemplary: minimal complexity, highly optimized, scalable, elegant algorithm design, fully maintainable, demonstrates advanced understanding of complexity and efficiency.

## Instructions

- Evaluate **only the code provided**.  
- Output **only** the numeric score (0-9).  
- Do not provide explanations, comments, or text beyond the score.  

**Code to Evaluate:**  """

system_prompts = {
    "quality": quality,
    "educational": educational,
    "complexity": complexity,
}
system_prompt = system_prompts[prompt_type]

print("Loading Model")
llm = LLM(model=model, download_dir=download_dir, tensor_parallel_size=1)
print("Model Loaded")

# Create a class to do batch inference.
def batch_inference(batch) -> Dict[str, list]:
    prompts = [[{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": content if len(content) < truncation_size else content[:truncation_size]}] for content in batch["code"]]
    outputs = llm.chat(prompts, 
                       sampling_params, 
                       use_tqdm=False,
                       )
    output_list = [output.outputs[0].text for output in outputs]
    batch["score"] = output_list
    return batch

# Iterate over files
print(input_data_path)
for i, file in enumerate(sorted(pathlib.Path(input_data_path).glob('*.parquet'))):
    # Break loop based on limit
    if i == int(limit):
        break
    print(f"-------------------------------------FILE: {file}-------------------------------------")
    # Load the Parquet file into a Hugging Face Dataset
    dataset = datasets.load_dataset('parquet', data_files=str(file))
    print(f"LOADED", flush=True)
    dataset_shard = dataset['train'].shard(index=GPU_id, num_shards=GPUS)
    print(f"SHARDED", flush=True)
    output_shard = dataset_shard.map(batch_inference, batched=True, batch_size=batch_size)
    print(f"MAPPED", flush=True)
    filename=os.path.basename(file)
    output_shard.to_parquet(f"{output_data_path}/shard_{GPU_id}_{filename}")

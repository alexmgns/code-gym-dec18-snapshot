import argparse
import json
import random

from typing import List, Dict, Type
from pydantic import BaseModel, create_model
from tester import Tester
from prompts.seeds import *
from utils.model import ModelInference, VLLMInference
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# Structured Decoding Schemas
class AugmentTask(BaseModel):
    prompt: str
    solution: str
    tests: str

class SimilarTask(BaseModel):
    prompt: str
    solution: str
    tests: str

class PromptTask(BaseModel):
    prompt: str
    solution: str
    tests: str

class TestTask(BaseModel):
    tests: str

# Code Data Generator
class CodeDataGenerator:
    """
    Handles code data generation using different strategies:
      1. Prompt-based generation
      2. Seed-based generation
      3. Augmentation of existing code samples
      4. Test generation for existing code
      5. Curriculum-based generation
    """

    def __init__(self, model: ModelInference, run_tests: bool = True):
        self.model = model
        self.run_tests = run_tests
        self.tester = Tester()

    @staticmethod
    def sample_seeds(n: int, k: int, sample_size: int) -> List[tuple[int, ...]]:
        """Sample combinations with replacement."""
        samples = set()
        while len(samples) < sample_size:
            comb = tuple(sorted(random.choices(range(1, n + 1), k=k)))
            samples.add(comb)
        return list(samples)

    # ----------------- Prompt-based -----------------
    def generate_from_prompt(self, prompt: str, num_samples: int) -> List[Dict]:
        results = self.model.generate(prompt, num_samples=num_samples)
        return results

    # ----------------- Seed-based -----------------
    def similar(self, seeds: List[Dict], combination_size: int, num_samples: int) -> List[Dict]:
        n = len(seeds)
        indices = self.sample_seeds(n, combination_size, n * num_samples)
        prompt = get_similar_data_prompt([seeds[i] for i in indices])
        results = self.model.generate(prompt)
        return results

    # ----------------- Augmentation -----------------
    def augment(self, seeds: List[str], num_samples: int) -> List[List[str]]:
        prompts = [get_augmentor_data_prompt(seed) for seed in seeds]
        results = self.model.generate(prompts, num_samples=num_samples)
        return results
    
    # ----------------- Tests -----------------
    def tests(self, seeds: List[str], num_samples: int) -> List[List[str]]:
        prompts = [get_test_data_prompt(seed) for seed in seeds]
        results = self.model.generate(prompts, num_samples=num_samples)
        return results


# ----------------- CLI -----------------
def create_parser():
    # Parent parser with shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--model", required=True, help="Model to use for generation")
    parent_parser.add_argument("--generate_tests", action="store_true", help="Run tests on generated samples")

    # Main parser
    parser = argparse.ArgumentParser(description="Code Data Generation CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Generation strategy")

    # ---- Prompt-based ----
    prompt_parser = subparsers.add_parser(
        "prompt",
        parents=[parent_parser],
        help="Prompt-based code generation"
    )
    prompt_parser.add_argument("--prompt", required=True, help="Prompt for generation")
    prompt_parser.add_argument("--num_samples", type=int, default=10)

    # ---- Seed-based ----
    similar_parser = subparsers.add_parser(
        "similar",
        parents=[parent_parser],
        help="Similar-based synthesis"
    )
    similar_parser.add_argument("--seed_path", required=True, help="Path to seed JSON file")
    similar_parser.add_argument("--combination_size", type=int, default=2)
    similar_parser.add_argument("--num_samples", type=int, default=1)

    # ---- Augmentation ----
    augment_parser = subparsers.add_parser(
        "augment",
        parents=[parent_parser],
        help="Augment existing code samples"
    )
    augment_parser.add_argument("--seed_path", help="Path to seed JSON file")
    augment_parser.add_argument("--num_samples", type=int, default=1)

    # ---- Tests ----
    tests_parser = subparsers.add_parser(
        "tests",
        parents=[parent_parser],
        help="Generate tests for code"
    )
    tests_parser.add_argument("--seed_path", help="Path to seed JSON file")
    tests_parser.add_argument("--num_samples", type=int, default=1)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Schema mapping
    schema_map: Dict[str, Type[BaseModel]] = {
        "prompt": PromptTask,
        "similar": SimilarTask,
        "augment": AugmentTask,
        "tests": TestTask,
    }
    if args.mode not in schema_map:
        raise ValueError(f"Unsupported mode '{args.mode}'")

    # Sampling parameters
    json_schema = {
        "type": "array",
        "items": schema_map[args.mode].model_json_schema(),
    }
    guided_decoding = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        temperature=0.7, max_tokens=512, stop=["<END>"], top_p=0.95, guided_decoding=guided_decoding
    )

    # Initialize model and generator
    model = VLLMInference(model_name=args.model, sampling_params=sampling_params)
    generator = CodeDataGenerator(model, run_tests=args.generate_tests)

    # Dispatch based on mode
    if args.mode == "prompt":
        samples = generator.generate_from_prompt(args.prompt, args.num_samples)
        print(samples)

    elif args.mode == "similar":
        with open(args.seed_path, "r") as f:
            seeds = json.load(f)
        similar = generator.similar(seeds, args.combination_size, args.num_samples)
        print(similar)

    elif args.mode == "augment":
        with open(args.seed_path, "r") as f:
            seeds = json.load(f)
        augmented = generator.augment(seeds, int(args.num_samples))
        print(augmented)

    elif args.mode == "tests":
        with open(args.seed_path, "r") as f:
            seeds = json.load(f)
        tests = generator.tests(seeds, int(args.num_samples))
        print(tests)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()

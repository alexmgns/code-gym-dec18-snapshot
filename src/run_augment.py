"""
A helper script that reads a verl-formatted dataset, augments it, and stores it.
"""

import json
import argparse
from typing import List, Callable

from transformation.assignment_unroll import UnrollAssignmentTransformation
from transformation.condition_unroll import UnrollConditionTransformation
from transformation.for_to_while import ForToWhileTransformation
from transformation.variable_rename import RandomizeVariableNames

def apply_transforms_to_example(example: dict, transforms: List[Callable[[str], str]]) -> dict:
   """
   Apply a sequence of transforms to the 'ground_truth' solutions of one example.
   """
   transformed_solutions = []
   for sol in example["reward_model"]["ground_truth"]:
      for t in transforms:
         sol = t(sol)
      transformed_solutions.append(sol)
   
   # Return a new example dict with transformed solutions
   new_example = example.copy()
   new_example["reward_model"] = example["reward_model"].copy()
   new_example["reward_model"]["ground_truth"] = transformed_solutions
   return new_example

def main(input_file: str, output_file: str, transform_names: List[str]):
   # Map transform names to actual callables
   available_transforms = { 
      "for_to_while": ForToWhileTransformation().transform_code,
      "condition_unroll": UnrollConditionTransformation().transform_code,
      "assignment_unroll" : UnrollAssignmentTransformation().transform_code,
      "randomize_var_names" : RandomizeVariableNames().transform_code
      # 4 transforms implemented so far. When there are lots more, this will be made more scalable.
   }

   # Make sure the user used valid transform names
   unknown = [name for name in transform_names if name not in available_transforms]
   if unknown:
      print("ERROR: Unknown transform(s):", ", ".join(unknown))
      print("Available transforms:")
      for name in sorted(available_transforms):
         print(f"  - {name}")
      raise SystemExit(1)
   
   

   transforms_to_apply = [available_transforms[name] for name in transform_names]

   # Load dataset
   with open(input_file, "r", encoding="utf-8") as f:
      dataset = json.load(f)

   # Apply transforms to each example
   transformed_dataset = [
      apply_transforms_to_example(example, transforms_to_apply)
      for example in dataset
   ]

   # Save transformed dataset
   with open(output_file, "w", encoding="utf-8") as f:
      json.dump(transformed_dataset, f, indent=2)
   print(f"Transformed dataset saved to {output_file}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Apply AST/code transforms to a verl-style dataset")
   parser.add_argument("--input", required=True, help="Path to input verl-dataset JSON")
   parser.add_argument("--output", required=True, help="Path to output transformed verl-dataset JSON")
   parser.add_argument("--transforms", nargs="+", required=True, help="List of transforms to apply")
   args = parser.parse_args()

   main(args.input, args.output, args.transforms)

"""
Example usage:

# Ex. run from src
python src/run_augment.py --input src/example_dataset.json \
    --output verl_dataset_transformed.json \
    --transforms assignment_unroll for_to_while
"""
# Code Gym
A project on improving models' coding capabilities. Primary dataset available at [Hugging Face](https://huggingface.co/datasets/newfacade/LeetCodeDataset). Hosted by the [SwissAI](https://www.swiss-ai.org/apertus) CodeGym team, this [ML4Science](https://epfml.github.io/cs433-2025/ml4science/) project was done as coursework for [CS-433 Machine Learning](https://epfml.github.io/cs433-2025/) at EPFL. 

## Note on Project Scope and Ownership

The following components **were not implemented by the student team**, but instead by the CodeGym team before onboarding, and are included for completeness or integration purposes only:

### Source directories
- `src/backend`
- `src/classifiers`
- `src/dataset` *(except for changes made to `utils` and `posttrain`)*
- `src/formatters`
- `src/masking`
- `src/prompts`
- `src/utils`

### Source files
- `src/synthesizer.py`
- `src/tester.py`

### Project root files and scripts
- `Dockerfile`
- `backend.sh` scripts
- `generation.sh` scripts

The **transformation framework and transformation implementations**, were developed as part of this project.

## Transformation

Point: Transform classes based on an abstract Transform class, providing a clear format for transformation application.
At: ```src/transformation/...```

### Available Code Transformations

- **`assignment_unroll`**  
  Rewrites chained or compound assignments into multiple simple assignment statements.  
  This makes data flow explicit and removes syntactic shortcuts.

- **`condition_unroll`**  
  Replaces chained comparisons in conditional expressions (e.g., in `if` and `while` statements) with multiple simple comparisons combined using logical operators.  
  This eliminates comparison shorthand and makes evaluation order explicit.

- **`for_to_while`**  
  Translates `for` loops into equivalent `while` loops by explicitly separating initialization, condition checks, and iteration updates.  
  This normalizes loop structure while preserving semantics.

- **`randomize_var_names`**  
  Replaces variable identifiers with randomly generated names while preserving program behavior.  
  This removes semantic information from naming and focuses analysis on structure rather than readability.


### Example as library:
```python
# Imports
dataset = LeetCodeDataset()
transform = ForToWhileTransformation()
transformed_dataset = transform.apply(dataset)
# Work with transformed Dataset
```
### Example as command:
```
python src/run_augment.py --input src/example_dataset.json \
    --output verl_dataset_transformed.json \
    --transforms assignment_unroll for_to_while
```
The dataset must have a `ground_truth` column with code. This column will be mutated by the code according to the transformation. An example dataset is included at `src/example_dataset.json` for convenience.

## Dependencies

Install dependencies with:
```
conda env create -f environment.yml
```

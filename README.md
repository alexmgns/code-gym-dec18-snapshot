# Code Gym
Improving models coding capabilities.



## Transformation

Point: Transform classes based on one class, providing a clear format for transformation application.
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

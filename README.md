# Code Gym
Improving models coding capabilities.

## Transformation
1. Point: Transform classes based on one class, providing a clear format for transformation application.
2. At: ```src/transformation/...```
3. Example as library:
```python
# Imports
dataset = LeetCodeDataset()
transform = ForToWhileTransformation()
transformed_dataset = transform.apply(dataset)
# Work with transformed Dataset
```
4. Example as command:
```
python src/run_augment.py --input src/example_dataset.json \
    --output verl_dataset_transformed.json \
    --transforms assignment_unroll for_to_while
```

## Dependencies

Install dependencies with:
```
conda env create -f environment.yml
```

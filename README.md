# Code Gym
Improving models coding capabilities.

## Transformation
1. Point: Transform classes based on one class, providing a clear format for transformation application.
2. At: ```src/transformation/...```
3. Example:
```python
# Imports
dataset = LeetCodeDataset()
transform = ForToWhileTransformation()
transformed_dataset = transform.apply(dataset)
# Work with transformed Dataset
```
## Dependencies

Install dependencies with:
```
uv venv --python 3.10
source ./venv/bin/activate
```

TODOs: 
1. test whether transformed code still passes tests
2. evaluate performance of the model that uses both original and transformed data 

## Examples
We divide examples into two:
1. Backend (for veRL integration): `backend.sh`
2. Data generation (Offline data generation): `generation.sh`

## TODOs:
### Future:
1. Generate/Use Datasets (Existing datasets, or synthetic ones)
2. Test them using SandboxFusion via Scheduler
3. Run in veRL

### Tasks
1. Integrate backend into veRL
2. Create a synthetic data pipeline generation (Dataset creation, Test creation, Test of tests => New Synthetic Pipeline)
  - Integrate OpenAPI model into the models.py => Easier for SGLang, VLLM serving etc...

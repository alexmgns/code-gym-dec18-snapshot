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


## Other componenets, not part of ML4Science project:
## Image
### Build
```bash
srun --account=infra01 --nodes=1 --time=4:00:00 --pty bash
podman build -t code-gym .
enroot import -x mount -o code-gym.sqsh podman://code-gym
```
Otherwise, image can be found at: `...`.
### Use
Create `vim $HOME/.edf/codegym.toml` and replace the variables:
```bash
image = "<SQSH_FILEPATH>"
mounts = ["/iopsstor/scratch/cscs/<USER_NAME>:/iopsstor/scratch/cscs/<USER_NAME>"]
workdir = "/iopsstor/scratch/cscs/<USER_NAME>"
writable=true
```
Then when submitting job use `--environmnet=code-gym`.

## Scheduler
1. Point: Multi-threaded serving of sandbox (fork of SandboxFusion)
2. At: ```src/backend/...```
3. Build custom image of sandbox:
```bash
git clone https://github.com/User3574/sandbox
cd sandbox 
# Edit the bash script build.sh with path to sandbox
sbatch build.sh
```
4. Load the image:
To use your own build image use:
```python
...
Sandbox(image=<PATH_TO_IMAGE>)
...
```

## Pretrain


## Posttrain



## Masking
1. Point: Serves as a masking utility for dataset creation
2. At: ```src/masking.py```
3. Example:
```python
# Load code here
dataset = CodeMaskDataset(List[<COMPLEXITY_METRICS>])
data = dataset.generate(code=code, masking_probability=<MASK_PROBABILITY>)
# Save data here
```
Where `<COMPLEXITY_METRICS>` are from `src/complexity.py` and `<MASK_PROBABILITY>` is determining how often we mask.

## Synthesizer
1. Point: Serves as synthetic data generator
2. At: ```src/synthesizer.py```
3. Example:
```bash
# Generate Training Samples
python src/generate_code.py \
  --mode <MODE> \
  --model <MODEL_NAME> 
  --kwargs ...
# Generate Tests
```
Where `<MODE>` specifies the type of data being generated and `<MODEL>` type of model to use for generation.

## Testers
1. Point: Test the tests and solutions, including coverage.
2. At: ```src/tester.py```
3. Example: 
```python
# Load code, tests here
tester = Tester()
result = tester.run(<PROGRAMMING_LANGUAGE>, code, tests)
# Save data here
```
Where `<PROGRAMMING_LANGUAGE>` is the lowercase programming language used.

## Dataset
1. Point: Dataset classes based on one class, providing evaluations as well as correct loading
2. At: ```src/dataset/...```
3. Example:
```python
# Imports 
dataset = get_dataset(<DATASET_NAME>)
# Work with Dataset
```

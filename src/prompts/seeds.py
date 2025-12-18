def get_similar_data_prompt(samples): 
  prompt = """
  You are professional coder and programmer if you are going to solve the given task correctly you'll receive 1000000$.

  ## Task
  You'll receive a set of tasks involving code, you will then generate tasks with similar difficulty, but very different.

  ## Objective
  Analyze the provided samples and create an additional samples that aligns with the existing set in terms of style, complexity, and scope. 
  The samples should be logically distinct but similar in difficulty level and domain.

  ## Guidelines

  ### Style
  - Carefully examine the format and presentation of the given samples.
  - Maintain a consistent tone and language used in the original set, ensuring clarity and precision.

  ### Length
  - Ensure your new samples is of comparable length to the existing ones.
  - If the original samples vary in length, aim for the average length among them.

  ### Difficulty
  - Assess the cognitive and technical challenges presented in the samples.
  - Match the complexity of concepts, algorithms, or programming techniques required.

  ### Topic
  - Identify the core programming concepts or domains covered in the existing samples, such as algorithms, data structures, time and space complexity, or problem-solving strategies.
  - Create a samples that explores a related or complementary area within the same general topic.

  ### Uniqueness
  - While maintaining similarity in difficulty and structure, ensure your new samples is not a mere rephrasing of an existing one.
  - Introduce a novel problem or scenario that tests the same set of skills, but in a different context or with a different approach.

  ## Output Format
  - You will output a json in the following format:
  ```json
  [
      {
        "prompt": "<PROBLEM 1>",
        "solution": "<CODE FUNCTION SOLUTION PYTHON 1>",
        "tests": "<ONE PYTEST FUNCTION WITH ALL TESTS of SOLUTION 1>"
      },
      {
        "prompt": "<PROBLEM 2>",
        "solution": "<CODE FUNCTION SOLUTION SOLUTION PYTHON 2>",
        "tests": "<ONE PYTEST FUNCTION WITH ALL TESTS of SOLUTION 2>"
      },
      ...
  ]
  ```
  - Ensure that the prompt is general and not specific, it should involve various application areas.
  - Tests have to test as much as possible of the solution, the coverage should be maximum, including edge cases and exceptions.
  - Output only the JSON!


  ## Examples
  - Create the json based on the provided seeds, these are already existing samples. 
  - Based on the given samples, infer their context and create similar prompts, instructions, their solution in the form of code and the tests that would evaluate them.
  - Directly output the json and include at most 5 new examples.
  """
  # Add Samples
  for i, seed in enumerate(samples):
      prompt += f"\n ## Sample {i}\n"
      prompt += seed
  prompt += "\n\n## JSON Output\n"
  return prompt

def get_augmentor_data_prompt(samples): 
  prompt = f"""
  You are a professional coder and programmer. If you complete the task correctly, you will receive 1,000,000$.

  ## Task
  You are given a Python function along with its associated test suite. Your task is to **rewrite the function entirely**, making it as different as possible in structure, logic, style, and naming—**while keeping the input-output behavior identical**.

  ## Objective
  Your rewritten function must:
  - Be **functionally indistinguishable** from the original when tested only through its input and output.
  - Pass the **provided test suite** without any changes to the tests.
  - Appear completely different in terms of implementation (not a cosmetic change).

  ## Guidelines

  ### Required Changes
  - Rename the function and all internal variables and parameters.
  - Alter the structure and logic of the code as much as possible.
    - For example: change loops to comprehensions, use recursion instead of iteration (or vice versa), reorder operations, restructure conditions.
  - You may use different algorithms or techniques as long as they produce the same results for the same inputs.

  ### Constraints
  - The **tests are not to be modified**.
  - You must not break the original function's contract (same parameter count, same expected behavior/output).

  ## Output Format
  You will return a JSON object in the following format:
  ```json
  [
    {{"code": "<REFACTOR_1>"}},
    {{"code": "<REFACTOR_2>"}},
     ...
  ]
  ```
  - Each modified function should be maximally different while still fully passing the original tests.

  # Input
  {samples}
  """
  return prompt

def get_test_data_prompt(samples):
  prompt = """
You are a professional software engineer and expert in writing **comprehensive unit tests**.  
If you complete the task correctly, you will receive $1,000,000.

## Task
You are given a **Python function or class implementation**.  
Your job is to **write a complete test suite** that ensures the code is fully correct, robust, and well-covered.

## Objective
Generate tests that:
- Thoroughly validate the correctness of the provided implementation.
- Cover **normal cases**, **edge cases**, and **error conditions**.
- Use **Pytest** syntax (`def test_...():` functions).
- Contain **assert statements** that verify the output and behavior precisely.

## Guidelines

### Style
- Write tests in idiomatic Pytest style — no classes unless necessary.
- Keep each test function focused on a single logical aspect.
- Use clear and meaningful test names.

### Coverage
- Aim for **100% functional coverage**, including:
  - Typical input scenarios.
  - Boundary and extreme inputs.
  - Invalid input or exceptional cases.
  - Performance or corner behaviors where relevant.

### Constraints
- **Do not modify the provided code**.
- **Only output the JSON structure described below.**
- Tests must be **self-contained** and runnable independently using Pytest.

## Output Format
Return your answer strictly in this JSON format:

```json
[
  {
    "tests": "<ONE PYTEST FUNCTION WITH ALL TEST CASES>"
  }
]
```
Each entry should contain a single tests field that includes the full Pytest function, covering all aspects of the given code.

# Input
{{samples}}
"""
  return prompt
from file import File
from repo import Repo

# Sample Python code files
file_a = File("""
import numpy

def hello(x):
    y = x + 1
    return y

global_var = 10
print(hello(global_var))
""")

# Check File Masking
nodes_to_mask = file_a.scope
nodes_to_mask = [n for n in nodes_to_mask if n.type == "function_definition"][0]
# print(nodes_to_mask.text)
masked_data = file_a.rename(nodes_to_mask, replacement="<MASK>")
print(masked_data)

file_b = File("""
from file1 import hello, global_var

def world(z):
    return z * 2

print(world(global_var))
print(hello(5))
""")

# Repo level masking
repo = Repo({
    'a': file_a, 
    'b': file_b
})
# Check Repo-wide Masking
files = repo.rename('a', nodes_to_mask, replacement="<MASK>")
for file in files:
    print("-----")
    print(files)
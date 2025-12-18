"""
This is an implementation of a transformation that adds LLVM intermediate representation
to the dataset, by concatenating it to the solution's code.

This implementation adheres to the Transformation interface, so it works nicely alongside
other transformations in this folder, but unfortunately it is not compatible with the Code Gym project's API
(due to the fact that it requires information from multiple columns to augment the code column).

Hence, a separate standalone script was used to augment the dataset for that purpose, starting directly 
from the LeetCode dataset. See ./legacy/gen_ir_dataset.py.
"""
import json
import subprocess
from tqdm import tqdm
import warnings

from transformation.base import Transformation
from dataset.utils import Dataset


# Imports are not part of the LeetCode solutions, so a bunch of commonly
# used imports are added to the beginning of every code.
IMPORTS = """
import collections
from collections import *
import itertools
from itertools import *
import functools
from functools import *
import heapq
from heapq import *
import math
from math import *

"""


def _ir_transform(solution, test_code=None, test_name=None, entry_point=None):
    code = IMPORTS + solution   
    code += "\n"
    if test_code is not None:
        code += test_code
        code += "\n"
        code += f"def main():\n    {test_name}({entry_point})\n    print('ok!')\nif __name__ == '__main__':\n    main()\n"
    try:
        p = subprocess.run(["codon", "build", "-release", "-llvm", "-o", "tmp.ll"], input=code, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        warnings.warn("[WARNING] Failed to augment example with IR due to compilation error.")
        warnings.warn(f"[WARNING] Error from compiler: \n {p.stderr}")
        return solution

    # For testing validity of output program
    # subprocess.run(["codon", "build", "-release", "-o", "tmp"], input=code, text=True)

    with open("tmp.ll", "r", encoding="UTF-8") as ftmp:
        llcode = ftmp.read()
    
    code += "\n\n\n" + llcode
    return code


class AugmentIRTransformation(Transformation):
    def apply(self, dataset: Dataset) -> Dataset:
        # Unfortunately, this is not compatible with the current API.
        # _ir_transform requires knowledge of additional columns, namely the
        # testing code. LeetCode solutions are just classes with a single method
        # implementing the solution. Compiling that in Codon results in an empty
        # result because there is no runnable code, just a class. For this reason,
        # the testing code must be added and invoked so that the compiler will not
        # optimize away the solution.
        # 
        # However, the example dataset used by run_augment.py does not suffer from this issue.

        # dataset.transfrom(_ir_transform)
        pass


    def transform_code(self, example: str) -> str:
        return _ir_transform(example)

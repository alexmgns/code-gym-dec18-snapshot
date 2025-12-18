"""
This is the legacy script that was used to augment the LeetCode dataset.
"""
import json
import subprocess
from tqdm import tqdm


def main():
    DATA_IN_PATH = "LeetCodeDataset-train.jsonl"
    DATA_OUT_PATH = "LeetCodeDataset-train-llvm.jsonl"

    failed = []

    with open(DATA_IN_PATH, "r", encoding="UTF-8") as fin, open(DATA_OUT_PATH, "w", encoding="UTF-8") as fout:
        for line in tqdm(fin.readlines()):
            current = json.loads(line)
            code = "from collections import *\n"
            code += "from itertools import *\n"
            code += "from functools import *\n"
            code += "import heapq\n"
            code += "from heapq import *\n"
            code += "import math\n"
            code += "from math import *\n"
            code += current["completion"]
            code += "\n"
            code += current["test"]
            code += "\n"
            code += f"def main():\n    check({current["entry_point"]})\n    print('ok!')\nif __name__ == '__main__':\n    main()\n"
            try:
                subprocess.run(["codon", "build", "-release", "-llvm", "-o", "tmp.ll"], input=code.encode("UTF-8"), check=True)
            except subprocess.CalledProcessError:
                failed.append(current)

            # To verify validity of generated code
            # subprocess.run(["codon", "build", "-release", "-o", "tmp"], input=code.encode("UTF-8"), check=True)
            with open("tmp.ll", "r", encoding="UTF-8") as ftmp:
                llcode = ftmp.read()
                current["llvm"] = llcode
            
            fout.write(json.dumps(current))
            fout.write("\n")
    
    print(f"Failed {len(failed)} examples")
    with open("wrong.json", "w", encoding="UTF-8") as f:
        json.dump(failed, f)


if __name__ == "__main__":
    main()
import subprocess
import tempfile

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List


# -------------------------
# Abstract Linter
# -------------------------
class Linter(ABC):
    @abstractmethod
    def lint(self, code: str):
        """Return (linted_code, status_message)."""
        pass

    @abstractmethod
    def lint_folder(self, path: str):
        """Return (linted_code, status_message)."""
        pass


# -------------------------
# Python Linter
# -------------------------
class PythonLinter(Linter):
    def lint(self, code: str):
        try:
            result = subprocess.run(
                ["ruff", "check", "--fix", "--stdin-filename", "temp.py", "-"],
                input=code,
                text=True,
                capture_output=True,
                check=True,
            )
            return result.stdout, "OK"
        except subprocess.CalledProcessError:
            return code, "LINTING_ERROR"

    def lint_folder(self, codes: List[str]):
        # Create temp directory
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_dir = Path(tmpdirname)
                paths = []

                # Write all code snippets to temp files
                for i, code in enumerate(codes):
                    file_path = tmp_dir / f"{i}.py"
                    file_path.write_text(code)
                    paths.append(file_path)

                result = subprocess.run(
                    ["ruff", "check", "--fix", tmpdirname],
                    capture_output=True,
                    check=False,
                )
                
                # Write all code snippets to temp files
                results = []
                for i, code in enumerate(codes):
                    file_path = tmp_dir / f"{i}.py"
                    results.append(file_path.read_text())
                return results, result.returncode
        except Exception as e:
            print(e.stderr)
            return codes, "LINTING_ERROR"


# Registry of linters
linters = {
    "python": PythonLinter(),
}

# # Example
# ps = PythonLinter()
# codes = [
#     "import os\ndef add(x,y):\nreturn x+y",
#     "import os\ndef add(x,y):\n     return x+y"
# ]
# print(codes)
# linted_codes, status = ps.lint_folder(codes)
# print(linted_codes)
# print(status)

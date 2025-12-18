import subprocess
import tempfile

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List


# -------------------------
# Abstract Formatter
# -------------------------
class Syntaxer(ABC):
    @abstractmethod
    def syntax(self, code: str):
        """Return (formatted_code, error_message)."""
        pass

    @abstractmethod
    def syntax_folder(self, codes: List[str]):
        """Return (formatted_code, error_message)."""
        pass


# -------------------------
# Python Formatter
# -------------------------
class PythonSyntaxer(Syntaxer):
    def syntax(self, code: str):
        try:
            result = subprocess.run(
                ["ruff", "format", "--stdin-filename", "temp.py", "-"],
                input=code,
                text=True,
                capture_output=True,
                check=True,
            )
            return result.stdout, "OK"
        except subprocess.CalledProcessError:
            return code, "FORMATTING_ERROR"

    def syntax_folder(self, codes):
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
                    ["ruff", "format", tmpdirname], 
                    capture_output=True, 
                    check=False
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


# Registry of available formatters
syntaxers = {
    "python": PythonSyntaxer(),
}

# # Example
# ps = PythonSyntaxer()
# codes = [
#     "import os\ndef add(x,y):\nreturn x+y",
#     "import os\ndef add(x,y):\n     return x+y"
# ]
# print(codes)
# formatted_codes, status = ps.syntax_folder(codes)
# print(formatted_codes)
# print(status)

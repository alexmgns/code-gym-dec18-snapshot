from file import File
from typing import Dict


class Repo:
    def __init__(self, files: Dict[str, File]):
        self.files = files

    def rename(self, masked_filename, masked_node, replacement="<MASK>"):
        results = {}
        # Mask across files
        for filename in self.files:
            file = self.files[filename]
            results[filename] = file.rename(masked_node, replacement)
        return results

from dataset.utils import Dataset
from abc import ABC

class Transformation(ABC):
   """Base class for all AST/code transformations."""

   supported_languages = ["python"]

   def apply(self, dataset: Dataset) -> Dataset:
      """
      lang = str(dataset.get_language())
      if lang not in self.supported_languages:
         raise NotImplementedError(
               f"'{lang}' transformations are not supported."
         ) 
      """
      return dataset # NOP; subclasses override

   def transform_code(self, example : str) -> str: 
      return example # NOP; subclasses override
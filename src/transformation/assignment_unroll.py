import ast
from transformation.base import Transformation
from dataset.utils import Dataset, extract_code
from transformation.utils import _collect_names, _unique_name

class AssignmentUnroller(ast.NodeTransformer):
   """
   Rewrites chained assignments like a = b = f(x)
   into:
      _temp = f(x)
      b = _temp
      a = _temp
   preserving evaluation order and side effects.
   """

   def __init__(self):
      super().__init__()
      self.used_names = set()

   def visit_Module(self, node: ast.Module):
      self.used_names = _collect_names(node)
      return self.generic_visit(node)

   def visit_Assign(self, node: ast.Assign):
      self.generic_visit(node)

      # Only handle multiple targets
      if len(node.targets) <= 1:
         return node

      # Evaluate rightmost value once
      temp_name = _unique_name("assign_tmp", self.used_names)
      temp_assign = ast.Assign(
         targets=[ast.Name(id=temp_name, ctx=ast.Store())],
         value=node.value
      )
      temp_val = ast.Name(id=temp_name, ctx=ast.Load())

      # Assign temp to each target in **reverse order** (right-to-left)
      assigns = []
      for target in reversed(node.targets):
         assigns.append(
               ast.Assign(targets=[target], value=temp_val)
         )

      # Return: temp assignment + all target assignments
      return [temp_assign] + assigns

def _transform_code(code: str) -> str:
   safe_cpy = code
   try:
      code = extract_code(code, include_block=False)
      tree = ast.parse(code)
      tree = AssignmentUnroller().visit(tree)
      ast.fix_missing_locations(tree)
      return ast.unparse(tree)
   except Exception as e:
      print(f"ERROR in assignment-unroll transform. {e}")
      print(f"Original datapoint: {safe_cpy}")
      return safe_cpy


class UnrollAssignmentTransformation(Transformation):
   """
   Rewrites chained assignments:
      a = b = f(x)
   into:
      b = f(x)
      a = b

   Preserves evaluation order and side-effect semantics.
   """
   def apply(self, dataset: Dataset) -> Dataset:
      return dataset.transform(_transform_code)
   
   def transform_code(self, example: str) -> str:
      return _transform_code(example)

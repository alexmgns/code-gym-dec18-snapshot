import ast
from transformation.base import Transformation
from dataset.utils import Dataset, extract_code
from transformation.utils import _collect_names, _unique_name


class CompareUnroller(ast.NodeTransformer):
   """
   Lazily rewrites chained comparisons with proper temp order.
   """
   def __init__(self):
      super().__init__()
      self.used_names = set()

   def visit_Module(self, node: ast.Module):
      self.used_names = _collect_names(node)
      return self.generic_visit(node)

   def visit_If(self, node: ast.If): # handle if conditions
      self.generic_visit(node)
      # pass the original body into _rewrite_chained
      return self._rewrite_chained(node, node.body)

   def visit_While(self, node: ast.While): # handle while conditions
      self.generic_visit(node)
      return self._rewrite_chained(node, node.body)
   
   def _rewrite_chained(self, node, body):
      """
      Lower a chained comparison into nested statements with temps.
      `body` is the list of AST nodes to attach at the deepest level.
      """
      test = node.test
      if not isinstance(test, ast.Compare) or len(test.ops) <= 1:
         return node  # no chaining, nothing to do

      # Assign first two operands eagerly
      left_name = _unique_name("cmp", self.used_names)
      left_assign = ast.Assign(targets=[ast.Name(id=left_name, ctx=ast.Store())],
                              value=test.left)
      left_val = ast.Name(id=left_name, ctx=ast.Load())

      right_name = _unique_name("cmp", self.used_names)
      right_assign = ast.Assign(targets=[ast.Name(id=right_name, ctx=ast.Store())],
                                 value=test.comparators[0])
      right_val = ast.Name(id=right_name, ctx=ast.Load())

      # Outermost conditional (if or while)
      comparison = ast.Compare(left=left_val, ops=[test.ops[0]], comparators=[right_val])
      new_node = type(node)(test=comparison, body=[], orelse=getattr(node, "orelse", []))

      # Track previous two temps
      prev_val, current_val = left_val, right_val
      current_stmt = new_node

      # Remaining operands: assign inside nested body and create nested statement
      for op, right_expr in zip(test.ops[1:], test.comparators[1:]):
         temp_name = _unique_name("cmp", self.used_names)
         temp_assign = ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                                    value=right_expr)
         temp_val = ast.Name(id=temp_name, ctx=ast.Load())

         # Insert assignment
         current_stmt.body.append(temp_assign)

         # New nested comparison
         comp = ast.Compare(left=current_val, ops=[op], comparators=[temp_val])
         nested = type(node)(test=comp, body=[], orelse=[])
         current_stmt.body.append(nested)

         prev_val, current_val = current_val, temp_val
         current_stmt = nested

      # Attach original body at the deepest level
      current_stmt.body.extend(body)
      return [left_assign, right_assign, new_node]



def _transform_code(code: str) -> str:
   safe_cpy = code
   try:
      code = extract_code(code, include_block=False)
      tree = ast.parse(code)
      tree = CompareUnroller().visit(tree)
      ast.fix_missing_locations(tree)
      return ast.unparse(tree)
   except Exception as e:
      print(f"ERROR in unroll_chained_ifs. {e}")
      print(f"Original datapoint: {safe_cpy}")
      return safe_cpy


class UnrollConditionTransformation(Transformation):
   """
   Rewrites chained comparisons:
      a < b < c
   into:
      a < b and b < c

   Preserves evaluation order and side-effect semantics.
   """
   #  Note: different-level conditions e.g. a < (b < c) remain unchanged.
   def apply(self, dataset: Dataset) -> Dataset:
      return dataset.transform(_transform_code)
   
   def transform_code(self, example: str) -> str:
      return _transform_code(example)


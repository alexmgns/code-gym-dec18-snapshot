"""
This is an implementation of an AST augmentation that replaces for loops with equivalent while loops.
For..else is not supported.
"""
import ast
from transformation.base import Transformation
from dataset.utils import Dataset, extract_code
from transformation.utils import _collect_names, _unique_name

class ForToWhileTransformer(ast.NodeTransformer):
    def __init__(self, used_names):
        super().__init__()
        self.used_names = used_names

    def visit_For(self, node):
        self.generic_visit(node)

        try:
            target = node.target.id

            # Determine if we are iterating over range() or any iterable
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                # range(...) case: wrap in list() to evaluate once
                iter_expr = ast.Call(func=ast.Name(id='list', ctx=ast.Load()), args=[node.iter], keywords=[])
            else:
                # arbitrary iterable: wrap in list() to materialize it
                iter_expr = ast.Call(func=ast.Name(id='list', ctx=ast.Load()), args=[node.iter], keywords=[])

            # Create temporary variables
            iter_name = _unique_name(f"{target}_iter", self.used_names)
            idx_name = _unique_name(f"{target}_idx", self.used_names)

            iter_assign = ast.Assign(
                targets=[ast.Name(id=iter_name, ctx=ast.Store())],
                value=iter_expr
            )

            idx_assign = ast.Assign(
                targets=[ast.Name(id=idx_name, ctx=ast.Store())],
                value=ast.Constant(0)
            )

            # While loop condition: _idx < len(_iter)
            loop_cond = ast.Compare(
                left=ast.Name(id=idx_name, ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                        args=[ast.Name(id=iter_name, ctx=ast.Load())],
                                        keywords=[])]
            )

            # Assign target = _iter[_idx]
            assign_target = ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=iter_name, ctx=ast.Load()),
                    slice=ast.Name(id=idx_name, ctx=ast.Load()),
                    ctx=ast.Load()
                )
            )

            # Increment _idx
            increment_idx = ast.AugAssign(
                target=ast.Name(id=idx_name, ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(1)
            )

            # While body: assign target + original body + increment
            while_body = [assign_target] + node.body + [increment_idx]
            while_node = ast.While(test=loop_cond, body=while_body, orelse=node.orelse)

            # Cleanup temporary variables (but not the target)
            cleanup = ast.Delete(targets=[
                ast.Name(id=iter_name, ctx=ast.Del()),
                ast.Name(id=idx_name, ctx=ast.Del())
            ])

            return [iter_assign, idx_assign, while_node, cleanup]
        
        except Exception as e:
            print("For transform failed. Aborting. Error message received:", e)
            return None

def _for_to_while(code):
    safe_cpy = code
    try:
        code = extract_code(code, include_block=False)
        tree = ast.parse(code)
        used_names = _collect_names(tree)
        new_tree = ForToWhileTransformer(used_names).visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception as e:
        print(f"ERROR in for-to-while transform. {e}")
        print(f"Original datapoint: {safe_cpy}")
        return safe_cpy  
        


class ForToWhileTransformation(Transformation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.transform(_for_to_while)
    
    def transform_code(self, example: str) -> str:
        return _for_to_while(example)


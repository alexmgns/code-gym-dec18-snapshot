"""
Utility methods for AST transformation methods.
"""

import ast

# Helper functions

def _collect_names(tree):
   """ 
   Returns a set of all variable names defined in the AST subtree.
   May be used for declaring a new variable without redefining it.
   """
   names = set()
   for node in ast.walk(tree):
      if isinstance(node, ast.Name):
         names.add(node.id)
      elif isinstance(node, ast.arg):
         names.add(node.arg)
   return names

def _unique_name(base, used_names):
   """
   Returns a unique variable name derived from `base` without colliding with any previous declarations.  
   """
   name = f"_{base}"
   if name not in used_names:
      used_names.add(name)
      return name
   counter = 0
   while name in used_names:
      counter += 1
      name = f"_{base}_{counter}"
   used_names.add(name)
   return name
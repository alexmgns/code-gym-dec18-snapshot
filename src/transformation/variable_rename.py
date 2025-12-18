import ast
import random
import string
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dataset.utils import Dataset, extract_code
from rope.base.project import Project
from rope.refactor.rename import Rename
from transformation.base import Transformation


def _collect_variable_names(source: str) -> Set[str]:
   """
   Collect names of variables that are written to (Store context).
   Does NOT include function or class names, only local variables,
   loop vars, assignment targets, etc.
   """
   tree = ast.parse(source)
   names: Set[str] = set()

   class _Visitor(ast.NodeVisitor):
      def visit_Name(self, node: ast.Name) -> None:
         if isinstance(node.ctx, ast.Store):
            names.add(node.id)
         self.generic_visit(node)

      def visit_arg(self, node: ast.arg) -> None:
         # function argument names are also variables in scope
         names.add(node.arg)
         self.generic_visit(node)

   _Visitor().visit(tree)
   return names


def _build_random_name_map(
        names: Set[str],
        rng: random.Random,
        length: int,
) -> Dict[str, str]:
   """
   Build a mapping from original variable name -> random new identifier.
   Deterministic order w.r.t. `names` for a fixed RNG state.
   """
   mapping: Dict[str, str] = {}
   used: Set[str] = set()

   def _random_identifier() -> str:
      # ensure first char is alphabetic or underscore
      first_chars = string.ascii_letters + "_"
      other_chars = string.ascii_letters + string.digits + "_"
      first = rng.choice(first_chars)
      rest = "".join(rng.choice(other_chars) for _ in range(max(0, length - 1)))
      return first + rest

   # sort names so order is deterministic
   for name in sorted(names):
      new_name = _random_identifier()
      while new_name == name or new_name in used:
         new_name = _random_identifier()
      mapping[name] = new_name
      used.add(new_name)

   return mapping


def _find_definition_positions(
        source: str,
        target_names: Set[str],
) -> Dict[str, List[Tuple[int, int]]]:
   """
   For each target name, collect (lineno, col_offset) positions where
   the name appears as a Name node in the AST.

   Rope can resolve identifiers from any occurrence (not just the
   "definition"), so we just need a stable position for that name.
   """
   tree = ast.parse(source)
   positions: Dict[str, List[Tuple[int, int]]] = {n: [] for n in target_names}

   class _Visitor(ast.NodeVisitor):
      def visit_Name(self, node: ast.Name) -> None:
         if node.id in positions:
            positions[node.id].append((node.lineno, node.col_offset))
         self.generic_visit(node)

   _Visitor().visit(tree)
   return positions


def _offset_from_line_col(source: str, lineno: int, col: int) -> int:
   """
   Convert (lineno, col) into character offset in `source` (0-based).
   """
   lines = source.splitlines(keepends=True)
   # lineno is 1-based
   if lineno < 1 or lineno > len(lines):
      raise ValueError(f"Invalid lineno {lineno} for source with {len(lines)} lines.")
   return sum(len(l) for l in lines[: lineno - 1]) + col

class RopeVariableRefactor:
   """
   Concrete refactor engine using rope to rename variables.
   """

   def __init__(self, project_root: Path):
      self._project_root = Path(project_root)
      self._project: Optional[Project] = None
      self._mod_path: Path = self._project_root / "mod.py"

   def setup(self) -> None:
      self._project_root.mkdir(parents=True, exist_ok=True)
      self._mod_path.touch(exist_ok=True)
      self._project = Project(str(self._project_root), ropefolder=None)

   def mutate_variables(self, request: str) -> str:
      rng = random.Random(42)
      source = extract_code(request, include_block=False)

      # write source to mod.py
      self._mod_path.write_text(source, encoding="utf-8")
      variable_names = _collect_variable_names(source)
      if not variable_names:
         return request

      # resource = self._project.get_file(str(self._mod_path))
      resource = self._project.get_file("mod.py")
      rename_map = _build_random_name_map(
         variable_names,
         rng=rng,
         length=6,
      )

      for old_name, new_name in rename_map.items():
         try:
            current_source = self._mod_path.read_text(encoding="utf-8")

            positions_for_name = _find_definition_positions(
               current_source,
               {old_name},
            ).get(old_name, [])

            if not positions_for_name:
               continue

            lineno, col = positions_for_name[0]
            offset = _offset_from_line_col(current_source, lineno, col)

            rename = Rename(self._project, resource, offset)
            change = rename.get_changes(new_name)
            self._project.do(change)

         except Exception as exc:
            continue

      return self._mod_path.read_text(encoding="utf-8")


def _transform_code(code: str) -> str:
   safe_cpy = code
   try:
      with tempfile.TemporaryDirectory() as tmpdir:
         engine = RopeVariableRefactor(str(tmpdir))
         engine.setup()
         mutated = engine.mutate_variables(code)
         return mutated

   except Exception as e:
      print(f"ERROR in RandomizeVariableNames transform. {e}")
      print(f"Original datapoint: {safe_cpy}")
      return safe_cpy


class RandomizeVariableNames(Transformation):
   def apply(self, dataset: Dataset) -> Dataset:
      return dataset.transform(_transform_code)

   def transform_code(self, example: str) -> str:
      return _transform_code(example)

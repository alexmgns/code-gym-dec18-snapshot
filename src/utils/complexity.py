import math
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import code_ast
from code_ast import ASTVisitor
from prompts.complexity import get_complexity_prompt


class Metric(ABC):
    """Abstract base class for all code complexity metrics."""

    @abstractmethod
    def compute(self, code: str, ast_tree: Optional[Any] = None) -> Any:
        """
        Compute the metric for the given code.

        Args:
            code (str): The source code to analyze.
            ast_tree (Optional[Any]): Pre-parsed AST of the code (optional).

        Returns:
            Any: The computed metric value(s).
        """
        pass


class LLMMetric(Metric):
    """
    Metric that uses a large language model (LLM) to evaluate code complexity.
    Expects a `model` parameter for generating responses.
    """

    def compute(self, code: Dict[str, Any], ast_tree: Optional[Any], model) -> None:
        data = {
            'code': code['code'],
            'masks': [code['masks'][mask_id]['masked'] for mask_id in code['masks']]
        }
        data_json = json.dumps(data)
        print(data_json)

        messages = [
            {"role": "system", "content": get_complexity_prompt()},
            {"role": "user", "content": data_json},
        ]
        response = model.generate([messages])
        print(response)


class LineCountMetric(Metric):
    """Counts lines of code, comments, and blank lines."""

    def compute(self, code: str, ast_tree: Optional[Any] = None) -> Dict[str, int]:
        total, comments, blanks = 0, 0, 0
        for line in code.splitlines():
            total += 1
            stripped = line.strip()
            if not stripped:
                blanks += 1
            elif stripped.startswith("#"):
                comments += 1
        return {"loc": total, "cloc": comments, "bloc": blanks}


class WordCountMetric(Metric):
    """Counts words in the source code."""

    def compute(self, code: str, ast_tree: Optional[Any] = None) -> int:
        return len(code.split())


class CyclomaticComplexityMetric(Metric):
    """Calculates cyclomatic complexity using AST traversal."""

    class CyclomaticComplexityVisitor(ASTVisitor):
        """AST visitor that accumulates complexity based on node types."""

        def __init__(self):
            self.complexity = 1

            # Complexity for compound statements
            compound_statement_complexity = {
                "class_definition": 4,
                "decorated_definition": 3,
                "for_statement": 3,
                "function_definition": 4,
                "if_statement": 2,
                "match_statement": 4,
                "try_statement": 5,
                "while_statement": 3,
                "with_statement": 3
            }

            # Complexity for simple statements
            simple_statement_complexity = {
                "assert_statement": 2,
                "break_statement": 1,
                "continue_statement": 1,
                "delete_statement": 2,
                "exec_statement": 4,
                "expression_statement": 2,
                "future_import_statement": 2,
                "global_statement": 2,
                "import_from_statement": 2,
                "import_statement": 1,
                "nonlocal_statement": 2,
                "pass_statement": 1,
                "print_statement": 1,
                "raise_statement": 2,
                "return_statement": 1,
                "type_alias_statement": 3
            }

            self.type_complexity = {
                **compound_statement_complexity,
                **simple_statement_complexity
            }

        def visit(self, node):
            if node.type in self.type_complexity:
                self.complexity += self.type_complexity[node.type]

    def compute(self, code: str, ast_tree: Optional[Any] = None) -> int:
        if ast_tree is None:
            ast_tree = code_ast.ast(code, lang="python")
        visitor = self.CyclomaticComplexityVisitor()
        ast_tree.visit(visitor)
        return visitor.complexity


class HalsteadMetric(Metric):
    """Computes Halstead complexity measures."""

    class HalsteadVisitor(ASTVisitor):
        """AST visitor that collects operators and operands for Halstead metrics."""

        def __init__(self):
            self.operators = set()
            self.operands = set()
            self.operator_count = 0
            self.operand_count = 0

        def visit(self, node):
            if node.type.endswith("_operator") or node.type == "assignment":
                self.operators.add(node.type)
                self.operator_count += 1

        def visit_identifier(self, node):
            self.operands.add(node.text)
            self.operand_count += 1

    def compute(self, code: str, ast_tree: Optional[Any] = None) -> Dict[str, float]:
        if ast_tree is None:
            ast_tree = code_ast.ast(code, lang="python")
        visitor = self.HalsteadVisitor()
        ast_tree.visit(visitor)

        n1, n2 = len(visitor.operators), len(visitor.operands)
        N1, N2 = visitor.operator_count, visitor.operand_count

        vocabulary_size = n1 + n2
        program_length = N1 + N2
        difficulty = (n1 / 2) * (N2 / n2) if n2 else 0
        volume = program_length * math.log2(vocabulary_size) if vocabulary_size else 0

        return {
            "Operators": n1,
            "Operands": n2,
            "Program Length": program_length,
            "Vocabulary Size": vocabulary_size,
            "Difficulty": difficulty,
            "Volume": volume
        }


class MaintainabilityIndexMetric(Metric):
    """Computes maintainability index based on cyclomatic complexity, Halstead volume, and LOC."""

    def compute(self, code: str, ast_tree: Optional[Any] = None, **kwargs) -> float:
        cc = kwargs.get("cc", 0)
        halstead = kwargs.get("halstead", {})
        loc = kwargs.get("loc", 0)
        volume = halstead.get("Volume", 0)

        if loc == 0:
            return 100.0

        mi = max(
            0.0,
            (171 - 5.2 * math.log(volume + 1) - 0.23 * cc - 16.2 * math.log(loc + 1)) * 100 / 171
        )
        return mi


class AllMetrics:
    """
    Convenience class to compute multiple code metrics at once.

    Metrics included:
    - Lines of code
    - Word count
    - Cyclomatic complexity
    - Halstead metrics
    - Maintainability index
    """

    def __init__(self):
        self.line_counter = LineCountMetric()
        self.word_counter = WordCountMetric()
        self.cc_calculator = CyclomaticComplexityMetric()
        self.halstead_calculator = HalsteadMetric()
        self.mi_calculator = MaintainabilityIndexMetric()

    def compute_all(self, code: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # Basic metrics
        line_counts = self.line_counter.compute(code)
        word_count = self.word_counter.compute(code)
        results.update(line_counts)
        results['woc'] = word_count

        # AST-based metrics
        try:
            ast_tree = code_ast.ast(code, lang="python")

            cc = self.cc_calculator.compute(code, ast_tree)
            halstead = self.halstead_calculator.compute(code, ast_tree)
            mi = self.mi_calculator.compute(code, ast_tree, cc=cc, halstead=halstead, loc=results['loc'])

            results['cc'] = cc
            results['halstead'] = halstead
            results['mi'] = mi
        except Exception:
            # Fail-safe defaults
            results['cc'] = 0
            results['halstead'] = None
            results['mi'] = None

        return results

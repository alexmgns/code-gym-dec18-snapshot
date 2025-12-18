import json

from tree_sitter_language_pack import get_parser, get_language
from tree_sitter import Node


class PythonLanguage:
    def __init__(self):
        self.grammar_path = "/iopsstor/scratch/cscs/rmachace/codegym/assets/grammar/python.json"
        self.parser = get_parser("python")
        self.language = get_language("python")
        self.grammar, self.types = self.load_grammar(self.grammar_path)
        self.symbol_table = {}

    def load_grammar(self, grammar_path: str):
        with open(grammar_path, "r") as f:
            grammar = json.load(f)

        allowed_bases = ["_compound_statement", "_simple_statement"]
        allowed_types = []

        for entry in grammar:
            if entry["type"] in allowed_bases:
                if "subtypes" in entry:
                    allowed_types.extend(subtype["type"] for subtype in entry["subtypes"])
                else:
                    allowed_types.append(entry["type"])
        # Always include identifiers for masking
        allowed_types.append("identifier")
        return grammar, allowed_types

    def get_scope(self, tree):
        stack = [tree.root_node]
        nodes = []
        while stack:
            node = stack.pop()
            if node.type in self.types:
                nodes.append(node)
            stack.extend(node.children)
        return nodes

    def parse(self, source: str):
        """Parse source and return bytes, AST, and full symbol usage table"""
        bytes_source = source.encode("utf8")
        ast = self.parser.parse(bytes_source)
        symbol_table = self.get_symbol_table(ast)
        scope_nodes = self.get_scope(ast)
        return bytes_source, ast, symbol_table, scope_nodes

    def get_symbol_table(self, tree):
        usage_dict = {}

        def walk(node: Node, scope_vars=None):
            if scope_vars is None:
                scope_vars = {}

            # Function definition
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    if name_node.text not in usage_dict:
                        usage_dict[name_node.text] = {"type": "function", "definition": node, "usages": []}
                # New scope for function body
                body = node.child_by_field_name("body")
                if body:
                    walk(body, scope_vars.copy())

            # Class definition
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    if name_node.text not in usage_dict:
                        usage_dict[name_node.text] = {"type": "class", "definition": name_node, "usages": []}
                body = node.child_by_field_name("body")
                if body:
                    walk(body, scope_vars.copy())

            # Variable assignment (single or multiple targets)
            elif node.type in ("assignment", "augmented_assignment"):
                target = node.child_by_field_name("left")
                value = node.child_by_field_name("right")
                # handle multiple targets (like a, b = ...)
                targets = [target] if target.type == "identifier" else target.children
                for t in targets:
                    if t.type == "identifier":
                        if t.text not in usage_dict:
                            usage_dict[t.text] = {"type": "variable", "definition": t, "usages": []}
                        scope_vars[t.text] = usage_dict[t.text]
                if value:
                    walk(value, scope_vars)

            # For loops: track loop variable
            elif node.type == "for_statement":
                target = node.child_by_field_name("left")
                if target and target.type == "identifier":
                    if target.text not in usage_dict:
                        usage_dict[target.text] = {"type": "variable", "definition": target, "usages": []}
                    scope_vars[target.text] = usage_dict[target.text]
                body = node.child_by_field_name("body")
                if body:
                    walk(body, scope_vars.copy())

            # With statements: track context manager variables
            elif node.type == "with_statement":
                items = node.children_by_field_name("optional_vars")
                for var in items:
                    if var.type == "identifier":
                        if var.text not in usage_dict:
                            usage_dict[var.text] = {"type": "variable", "definition": var, "usages": []}
                        scope_vars[var.text] = usage_dict[var.text]
                body = node.child_by_field_name("body")
                if body:
                    walk(body, scope_vars.copy())

            # Imports
            elif node.type == "import_statement":
                for named_node in node.children_by_field_name("name"):
                    usage_dict[named_node.text] = {"type": "import", "definition": '*', "usages": []}

            elif node.type == "import_from_statement":
                module_node = node.children_by_field_name("module_name")[0]
                names_node = node.children_by_field_name("name")
                for n in names_node:
                    usage_dict[n.text] = {"type": "import", "definition": module_node.text, "usages": []}

            # Function calls
            elif node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node and func_node.type == "identifier":
                    if func_node.text not in usage_dict:
                        usage_dict[func_node.text] = {"type": "function", "definition": None, "usages": []}
                    usage_dict[func_node.text]["usages"].append(node)

            # Track variable usages (identifiers)
            elif node.type == "identifier":
                if node.text in scope_vars:
                    scope_vars[node.text]["usages"].append(node)

            # Recurse
            for child in node.children:
                walk(child, scope_vars.copy())

        walk(tree.root_node)
        return usage_dict

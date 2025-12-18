from utils import PythonLanguage
from tree_sitter import Node


class File:
    def __init__(self, source: str):
        self.source = source
        self.language = PythonLanguage()
        self.bytes, self.tree, self.symbols, self.scope = self.language.parse(source)

    def rename(self, masked_node, replacement):
        name_node = masked_node.child_by_field_name("name")
        code_bytes = self.bytes[:]

        # If we have named node
        if name_node:
            symbol = name_node.text
            info = self.symbols.get(symbol)
            print(info)
            if info:
                definition = info["definition"]
                # Locally defined
                if isinstance(definition, Node):
                    nodes_to_mask = [definition] + info.get("usages", [])
                # Globally defined (imported)
                else:
                    nodes_to_mask = info.get("usages", [])
                    # TODO: SORT THE IMPORT REPLACEMENT IN THE NAME (ie. replace the imported name)
                nodes = sorted(nodes_to_mask, key=lambda n: n.start_byte, reverse=True)
                for n in nodes:
                    code_bytes = (code_bytes[:n.start_byte] + replacement.encode("utf8") + code_bytes[n.end_byte:])
        
        # Unnamed node
        else:
            raise ValueError("Node does not have a name field")
        
        return code_bytes.decode("utf8")

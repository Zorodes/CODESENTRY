"""
AST-aware chunking for code retrieval.

Naive RAG chunks code by fixed line/token windows, which routinely splits a
function in half and destroys retrieval quality. This chunker uses tree-sitter
to walk the actual syntax tree and extract whole functions/classes/methods as
chunks, each tagged with file path, name, and line range so citations in the
final review can point at exact locations.

Unsupported languages fall back to whole-file chunking (still better than
mid-function splits, since files are small enough to fit in embeddings.)
"""

from dataclasses import dataclass, asdict
from tree_sitter import Language, Parser

import tree_sitter_python as ts_python
import tree_sitter_javascript as ts_javascript
import tree_sitter_typescript as ts_typescript
import tree_sitter_go as ts_go
import tree_sitter_java as ts_java
import tree_sitter_rust as ts_rust
import tree_sitter_ruby as ts_ruby
import tree_sitter_c as ts_c
import tree_sitter_cpp as ts_cpp

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
}

# Each language package exposes a `.language()` capsule wrapped into a
# tree_sitter.Language. tsx/typescript package exposes two grammars.
_LANGUAGE_BUILDERS = {
    "python": lambda: Language(ts_python.language()),
    "javascript": lambda: Language(ts_javascript.language()),
    "typescript": lambda: Language(ts_typescript.language_typescript()),
    "tsx": lambda: Language(ts_typescript.language_tsx()),
    "go": lambda: Language(ts_go.language()),
    "java": lambda: Language(ts_java.language()),
    "rust": lambda: Language(ts_rust.language()),
    "ruby": lambda: Language(ts_ruby.language()),
    "c": lambda: Language(ts_c.language()),
    "cpp": lambda: Language(ts_cpp.language()),
}

# Node types that represent a "chunkable unit" per language. Tree-sitter grammars
# name nodes differently, so this map has to be maintained per language.
CHUNK_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"class_declaration", "method_declaration", "interface_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item"},
    "ruby": {"method", "class", "module"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}

_LANGUAGE_CACHE = {}
_PARSER_CACHE = {}


@dataclass
class CodeChunk:
    file_path: str
    language: str
    chunk_type: str      # "function" | "class" | "method" | "file"
    name: str            # function/class name, or filename for whole-file fallback
    start_line: int
    end_line: int
    code: str

    def to_dict(self):
        return asdict(self)


def _get_parser(language: str) -> Parser:
    if language not in _PARSER_CACHE:
        if language not in _LANGUAGE_CACHE:
            builder = _LANGUAGE_BUILDERS.get(language)
            if builder is None:
                raise ValueError(f"No grammar builder registered for '{language}'")
            _LANGUAGE_CACHE[language] = builder()
        _PARSER_CACHE[language] = Parser(_LANGUAGE_CACHE[language])
    return _PARSER_CACHE[language]


def _node_name(node, source: bytes) -> str:
    """Best-effort extraction of the identifier name for a definition node."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return "<anonymous>"


def _walk(node, language: str, source: bytes, file_path: str, chunks: list, seen_ranges: set):
    chunk_types = CHUNK_NODE_TYPES.get(language, set())

    if node.type in chunk_types:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        key = (start_line, end_line)
        if key not in seen_ranges:
            seen_ranges.add(key)
            code = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            kind = "class" if "class" in node.type else (
                "method" if "method" in node.type else "function"
            )
            chunks.append(CodeChunk(
                file_path=file_path,
                language=language,
                chunk_type=kind,
                name=_node_name(node, source),
                start_line=start_line,
                end_line=end_line,
                code=code,
            ))
        # Still recurse into class bodies to pull out individual methods too.

    for child in node.children:
        _walk(child, language, source, file_path, chunks, seen_ranges)


def chunk_file(file_path: str, content: str) -> list[CodeChunk]:
    ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
    language = EXTENSION_TO_LANGUAGE.get(ext)

    if not language:
        return [_whole_file_chunk(file_path, content)]

    try:
        parser = _get_parser(language)
        source_bytes = content.encode("utf-8", errors="replace")
        tree = parser.parse(source_bytes)
        chunks: list[CodeChunk] = []
        seen_ranges: set = set()
        _walk(tree.root_node, language, source_bytes, file_path, chunks, seen_ranges)

        if not chunks:
            return [_whole_file_chunk(file_path, content, language=language)]
        return chunks
    except Exception as e:
        print(f"Chunking failed for {file_path} ({e}); falling back to whole-file.")
        return [_whole_file_chunk(file_path, content, language=language)]


def _whole_file_chunk(file_path: str, content: str, language: str = "unknown") -> CodeChunk:
    line_count = content.count("\n") + 1
    return CodeChunk(
        file_path=file_path,
        language=language,
        chunk_type="file",
        name=file_path.split("/")[-1],
        start_line=1,
        end_line=line_count,
        code=content,
    )

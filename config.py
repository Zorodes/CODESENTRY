import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_BASE = "https://api.github.com"

if not GITHUB_TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN not set. Create a .env file with GITHUB_TOKEN=ghp_xxx "
        "(personal access token, no special scopes needed for public repos)."
    )

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# File extensions the chunker knows how to parse with tree-sitter.
# Extend this as you add more languages in chunker.py's LANGUAGE_MAP.
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs", ".rb", ".c", ".cpp", ".h",
}

# Directories to skip entirely during ingestion (build artifacts, deps, etc.)
SKIP_DIRS = {
    "node_modules", "dist", "build", ".git", "venv", ".venv", "__pycache__",
    "vendor", "target", ".next", "coverage", "migrations",
}

MAX_FILE_SIZE_BYTES = 200_000  # skip huge generated/minified files

#!/usr/bin/env bash
set -euo pipefail

# Check uv is installed
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment .venv ..."
uv venv --python 3.10

echo "Installing dependencies ..."
uv sync --extra dev

echo "Installing pre-commit hooks ..."
.venv/bin/pre-commit install

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
echo "  # or: direnv allow  (if direnv is installed)"

#!/bin/bash
# Resolve the project directory from this script's own location (works with spaces /
# trailing spaces in the path, e.g. "Minor Project ").
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
else
    echo "⚠️  .venv not found in $SCRIPT_DIR — using system python" >&2
fi
echo ""
echo "🚀 Starting DSLT Photo Restoration Server..."
echo ""
python app.py

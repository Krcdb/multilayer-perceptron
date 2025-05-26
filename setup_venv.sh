#!/bin/bash

VENV_NAME=".venv"

python3 -m venv "$VENV_NAME"

if [ $? -ne 0 ]; then
  echo "❌ Failed to create virtual environment."
  exit 1
fi

source "$VENV_NAME/bin/activate"

pip install --upgrade pip

echo "✅ Virtual environment '$VENV_NAME' created and activated."
echo "📦 You can now install packages using pip."

#!/bin/bash

# Name of the virtual environment folder
VENV_NAME=".venv"

# Create virtual environment
python3 -m venv "$VENV_NAME"

# Check if creation was successful
if [ $? -ne 0 ]; then
  echo "❌ Failed to create virtual environment."
  exit 1
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"

# Upgrade pip inside the virtual environment
pip install --upgrade pip

echo "✅ Virtual environment '$VENV_NAME' created and activated."
echo "📦 You can now install packages using pip."

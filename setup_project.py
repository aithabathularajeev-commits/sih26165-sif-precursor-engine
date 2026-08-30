#!/usr/bin/env python3
"""
setup_project.py - Initialize project directories and structure
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Create necessary directories
dirs_to_create = [
    REPO_ROOT / "src" / "models",
    REPO_ROOT / "src" / "evaluation",
    REPO_ROOT / "models",
    REPO_ROOT / "outputs",
]

for dir_path in dirs_to_create:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created/verified: {dir_path}")

print("\nProject directories initialized successfully.")

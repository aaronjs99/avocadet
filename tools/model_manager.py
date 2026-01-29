#!/usr/bin/env python3
"""
Model Manager for Avocadet

Usage:
    python3 tools/model_manager.py list
    python3 tools/model_manager.py set <filename>
"""

import os
import sys
import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
SYMLINK_NAME = "best.pt"


def list_models():
    if not MODELS_DIR.exists():
        print(f"Error: Models directory not found at {MODELS_DIR}")
        return

    print(f"Models in {MODELS_DIR}:")
    print("-" * 40)

    current_target = None
    symlink_path = MODELS_DIR / SYMLINK_NAME
    if symlink_path.is_symlink():
        current_target = symlink_path.resolve().name

    files = sorted(
        [f for f in MODELS_DIR.iterdir() if f.is_file() and f.name != "README.md"]
    )

    for f in files:
        status = " "
        if f.name == SYMLINK_NAME:
            continue

        if f.name == current_target:
            status = "*"
        elif f.name == SYMLINK_NAME:
            continue  # Should be handled by is_symlink check usually, but just in case

        print(f"[{status}] {f.name}")

    print("-" * 40)
    if current_target:
        print(f"* 'best.pt' currently points to: {current_target}")
    else:
        print(f"No 'best.pt' symlink active.")


def set_best(filename):
    target_path = MODELS_DIR / filename
    if not target_path.exists():
        print(f"Error: Model '{filename}' not found in {MODELS_DIR}")
        return

    symlink_path = MODELS_DIR / SYMLINK_NAME

    # Remove existing symlink or file
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    try:
        # Create relative symlink
        os.symlink(filename, symlink_path)
        print(f"Successfully linked {SYMLINK_NAME} -> {filename}")
    except OSError as e:
        print(f"Error creating symlink: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage Avocadet AI Models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available models")

    set_parser = subparsers.add_parser("set", help="Set the active 'best.pt' model")
    set_parser.add_argument("filename", help="Filename of the model to use")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "set":
        set_best(args.filename)


if __name__ == "__main__":
    main()

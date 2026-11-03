#!/usr/bin/env python3
"""
Batch Multilingual Audio Generator for IMS-Toucan HD Modular TTS
Generates sample .wav audio for all 21 supported languages with moods & emotions
using the single locked voice (`models/universal_speaker.bin`) into `./output/`.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import generate_all_outputs, OUTPUT_DIR  # noqa: E402


def main() -> None:
    generate_all_outputs(OUTPUT_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""interactive_demo.py — A fully interactive, styled presentation script."""

import sys
import time
from pathlib import Path

# Add src to path so we can use the pipeline directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt2model.parsing import parse_prompt
from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
from prompt2model.models import recommend_model_name
from prompt2model.augmentations import build_augmentation_plan

# ANSI Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def typing_print(text, speed=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

def header(text):
    print("\n" + "=" * 60)
    print(f"  {BOLD}{BLUE}{text}{RESET}")
    print("=" * 60)
    time.sleep(0.5)

def step(text):
    print(f"\n{BOLD}{YELLOW}→ {text}{RESET}")
    time.sleep(0.8)

def success(text):
    print(f"{BOLD}{GREEN}✓ {text}{RESET}")
    time.sleep(0.3)

def main():
    header("PROMPT2MODEL: INTERACTIVE VISION FACTORY")
    
    print(f"{BOLD}Please enter your natural language prompt below:{RESET}")
    user_prompt = input(f"{BOLD}{GREEN}>> {RESET}")
    
    if not user_prompt.strip():
        print(f"{RED}Prompt cannot be empty!{RESET}")
        return

    step("Parsing Prompt & Planning Data Pipeline...")
    
    # We pass a dummy dataset config to the parser for the demo
    dataset = DatasetConfig(root="data/dummy", format=DatasetFormat.IMAGEFOLDER, image_size=160)
    
    # Actually parse the user's live prompt!
    config = parse_prompt(user_prompt, dataset)
    
    time.sleep(1)
    success(f"Task identified: {config.task.value.upper()}")
    
    # Autonomously recommend the backbone based on their parsed constraints
    model_name = recommend_model_name(config.task, config.constraints.priority)
    success(f"Recommended Backbone: {model_name.upper()}")
    
    # Dynamically build the augmentation strategy based on parsed environment descriptors
    aug_plan = build_augmentation_plan(config.data_context.environment_tags, config.task)
    if aug_plan.operations:
        aug_names = ", ".join(aug_plan.summary()).upper()
        success(f"Augmentation Strategy: {aug_names}")
    else:
        success("Augmentation Strategy: STANDARD_ONLY")

    step("Generating Training Requirements...")
    time.sleep(1)
    
    CYAN = "\033[36m"
    labels = [l.name for l in config.labels]
    
    req_format = "COCO Detection Format (.json)" if config.task.value == "detection" else "ImageFolder Format (Class-named subdirectories)"
    print(f"\n{BOLD}{YELLOW}To physically train this {config.task.value.upper()} model, you must provide:{RESET}")
    print(f"1. A dataset formatted in {BOLD}{GREEN}{req_format}{RESET}.")
    if config.task.value == "detection":
        print(f"2. Bounding box annotations for the following requested labels: {BOLD}{GREEN}{labels}{RESET}")
    else:
        print(f"2. Folders named exactly after the requested classes: {BOLD}{GREEN}{labels}{RESET}")
    print(f"3. A machine with at least 1 GPU (or a strong CPU) to execute the Ray Tune HPO.")

    header("HOW TO EXECUTE")
    
    print(f"{BOLD}Run the following command in your terminal to begin the actual training pipeline:{RESET}\n")
    
    cmd = (f"python -m prompt2model.cli run \\\n"
           f"    --prompt \"{user_prompt}\" \\\n"
           f"    --dataset-root \"/path/to/your/dataset\" \\\n"
           f"    --dataset-format \"{'coco' if config.task.value == 'detection' else 'imagefolder'}\" \\\n"
           f"    --enable-hpo")
    
    print(f"{CYAN}{cmd}{RESET}\n")
    
    typing_print(f"{BOLD}{GREEN}Once training completes, the model will be exported to 'output/model.onnx' for autonomous edge inference!{RESET}", 0.04)
    print("\n")

if __name__ == "__main__":
    main()

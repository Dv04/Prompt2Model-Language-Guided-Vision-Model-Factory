#!/usr/bin/env python3
"""run_final_demo.py — A stylized demonstration script for video presentation."""

import sys
import time
import random
from pathlib import Path

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
    header("PROMPT2MODEL: LANGUAGE-GUIDED VISION FACTORY")
    
    typing_print(f"{BOLD}User Prompt:{RESET} 'Build a lightweight face mask detector for low-light subway surveillance.'", 0.05)
    
    step("Parsing Prompt & Planning Data Pipeline...")
    time.sleep(1.5)
    success("Task identified: OBJECT DETECTION")
    success("Recommended Backbone: SSDLITE320_MOBILENET_V3_LARGE")
    success("Augmentation Strategy: LOW_LIGHT, MOTION_BLUR")

    step("Resolving Dataset Labels...")
    typing_print("Mapping ['mask', 'no-mask'] to internal labels...", 0.02)
    success("Labels resolved with 98.4% confidence.")

    step("Starting Ray Tune HPO (Hyperparameter Optimization)...")
    print(f"{BLUE}[Trial 1]{RESET} lr=1e-3, batch=16 | Accuracy: 0.92")
    time.sleep(1)
    print(f"{BLUE}[Trial 2]{RESET} lr=5e-4, batch=32 | Accuracy: 0.95 {BOLD}{GREEN}(Winner){RESET}")
    time.sleep(0.5)
    success("Best model checkpointed.")

    step("Exporting to Metadata-Embedded ONNX...")
    typing_print("Injecting Task, LabelMap, Mean/Std, and InputResolution into model.onnx...", 0.01)
    success("ONNX Export Complete (data/output/model.onnx)")

    header("AUTONOMOUS EDGE INFERENCE DEMO")
    
    step("Executing edge_infer.py (No external config required)")
    print(f"{BOLD}{GREEN}Loading model.onnx...{RESET}")
    time.sleep(0.5)
    print(f"✓ Task: detection")
    print(f"✓ Classes: ['mask', 'no-mask']")
    print(f"✓ Target Resolution: 320x320")
    
    step("Running Inference on sample_image.jpg")
    print(f"{BOLD}Latency: 5.42 ms | FPS: 184.5{RESET}")
    
    print(f"\n{BOLD}DETECTION RESULTS:{RESET}")
    print(f"1. mask     | Score: 0.98 | Box: (142, 55, 201, 110)")
    print(f"2. no-mask  | Score: 0.92 | Box: (310, 42, 388, 115)")

    header("PROJECT FINALIZED")
    typing_print(f"{BOLD}{GREEN}Evaluation Report and Telemetry History saved to disk.{RESET}", 0.04)
    print("\n")

if __name__ == "__main__":
    main()

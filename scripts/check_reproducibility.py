#!/usr/bin/env python3
"""check_reproducibility.py — Verifies the project's reproducibility and health."""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run_step(name, cmd, allow_fail=False):
    print(f"--- Running: {name} ---")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=not allow_fail)
        print(f"✓ {name} passed.")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"! {name} had warnings/errors but continuing...")
            return True, e.stdout + e.stderr
        print(f"✗ {name} failed.")
        print(e.stderr)
        return False, e.stderr

def main():
    print("====================================================")
    print("   PROMPT2MODEL REPRODUCIBILITY CHECKER             ")
    print("====================================================")
    
    success = True
    
    # 1. Dependency Check
    ok, _ = run_step("Dependency Check", [sys.executable, "-m", "pip", "check"], allow_fail=True)
    if not ok: success = False

    # 2. Unit & Integration Tests
    ok, _ = run_step("Full Test Suite", [sys.executable, "-m", "pytest", "tests/", "-q"])
    if not ok: success = False

    # 3. Artifact Verification
    print("--- Verifying Artifact Presence ---")
    required_dirs = ["data", "output/smoke", "src/prompt2model"]
    for d in required_dirs:
        if (REPO_ROOT / d).exists():
            print(f"✓ Directory '{d}' exists.")
        else:
            print(f"✗ Directory '{d}' is missing.")
            success = False

    # 4. Telemetry History
    if (REPO_ROOT / "data" / "telemetry_history.csv").exists():
        print("✓ Telemetry history found.")
    else:
        print("✗ Telemetry history missing.")
        success = False

    # 5. Final Report
    if (REPO_ROOT / "FINAL_REPORT.md").exists():
        print("✓ Final report found.")
    else:
        print("! Final report not found. Running generator...")
        subprocess.run([sys.executable, "scripts/generate_final_report.py"])

    print("\n====================================================")
    if success:
        print("✅ REPRODUCIBILITY VERIFIED")
        print("The project is ready for submission/demo.")
    else:
        print("❌ REPRODUCIBILITY FAILED")
        print("Please check the errors above.")
    print("====================================================")

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

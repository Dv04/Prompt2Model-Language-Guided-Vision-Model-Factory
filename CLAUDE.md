# Prompt2Model - agent guide

## Identity

A compiler from a plain-language intent ("watch dock 3 for a person without
a helmet, <100 ms, prioritise recall") to a distilled, quantized,
TensorRT-deployable, calibrated edge vision model - B1 of the 12-product
Dhi research program (the language-edge-model factory). Spec:
`aixavier/docs/RESEARCH/B_edge_vlm/B1_language_edge_model_factory.md`.

## Run the tests

```bash
.venv/bin/python -m pytest
```

No `PYTHONPATH` needed - the package is installed editable
(`.venv/bin/python -m pip install -e .`, `--system-site-packages` venv so it
can still see the machine's torch build).

## Architecture

`src/prompt2model/`:
- `parsing.py` - deterministic regex/keyword prompt → typed `PipelineConfig` (the offline-first default, dependency-free).
- `planner.py` - LLM planner front end; overlays extracted fields onto the regex parse, never replaces it.
- `config.py` - `ModelConstraints` / `PipelineConfig` schema (priority, latency, power, accuracy floor, param cap).
- `label_resolution.py` - CLIP-based label resolution with lexical fallback (lazy-loaded CLIP).
- `data.py` - classification + COCO-style detection dataset loaders, synthetic toy-data generators.
- `augmentations.py` - torchvision-native augmentation injection.
- `models.py` - lightweight model registry; `recommend_model()` applies constraints (a stated constraint only ever narrows the choice down, never up).
- `training.py` - training loops (Adam/CE) for classification + detection.
- `hpo.py` / `tuning.py` - Optuna-based hyperparameter search.
- `compression.py` - distill → quantize → accuracy-floor gate (`decide_gate` / `apply_compression`).
- `calibration.py` - temperature scaling + split-conformal abstain threshold, embedded in ONNX metadata.
- `targets.py` - pluggable deployment targets (onnxruntime default, tensorrt/jetson via `trtexec` or emitted build script).
- `exporting.py` - ONNX export with metadata injection.
- `edge_inference.py` - zero-config `EdgeModel` runtime; applies calibration + abstain at inference time.
- `evaluation.py` - metric harness.
- `reporting.py` - markdown evaluation report generation.
- `flywheel.py` - `HardCaseStore`: bounded capture of abstained/low-confidence frames for retraining (capture only, no labeling/active-learning).
- `ablation.py` - language-to-config ablation suite.
- `telemetry.py` - run telemetry.
- `pipeline.py` - `Prompt2ModelFactory`, the end-to-end orchestrator.
- `cli.py` - `prompt2model` CLI (`run`, `smoke-test`, `generate-toy-data`, `flywheel`).

## Hard rules

1. **Never fake a trained model.** GPU-needing work (larger backbones, real distillation runs at scale, TRT engine builds off-device) is recipes + CPU-verified math + eval protocols only, until compute exists (board item AC4). The synthetic-data smoke path is a validation tool, not a substitute for a real training claim.
2. **The accuracy-floor REFUSAL gate in `compression.py` is a feature, never soften it.** The binding floor is the user's explicit `constraints.accuracy_floor` when one was stated: the user contract is authoritative, whether looser or stricter. Only when no absolute floor was stated does the relative retention default `accuracy_floor_relative * baseline_accuracy` (default 0.98) apply. If a compressed artifact can't hold the binding floor, the factory refuses to ship it and keeps the uncompressed one. Never make the gate "try harder" to pass, never bypass it, never silently ship a failing artifact.
3. **Every module gets tests against synthetic ground truth as it's built.** Full `pytest` suite must be green before any commit. This repo's `main` is bootstrapped from a course project - new work goes on branches with draft PRs, not direct pushes.
4. **No Claude/Anthropic attribution in commits or PRs.** Brand is Dhi. This ships commercially.
5. **Local LLM is `qwen3.5:2b-q4_K_M` via Ollama's NATIVE `/api/chat`, not the OpenAI-compat layer, with `think:false`.** `planner.py`'s `api_flavor="ollama"` path already does this correctly (measured 8.6 s vs the OpenAI-compat layer stalling); the `/no_think` prompt-level marker is an OPT-IN fallback for OpenAI-flavor servers only - it actively derails qwen3.5 native chat and must stay off (`no_think_prompt_marker=False`) for the ollama flavor. Never regress this by routing qwen3.5 through the compat layer or injecting `/no_think` into its prompt.

## Current state + next steps

Week-4+ integrated prototype: 23 test files / 119 tests, all tracks
(Dev/Venkata/Madhuvani + B1 factory-compiler) landed. Proof run recorded in
PR #4's description; PR #4 is open on this repo (not yet merged to main).
Classification path is fully validated end-to-end (prompt → config →
training → metrics → ONNX → report); detection export is a best-effort
wrapper, not yet the fully validated path. Repo transfer to
`DHI-Technologies-Inc` is queued for after PR #4 merges (board item AC5).
Next: knowledge distillation and quantization deepen past the current
INT8/dynamic baseline, generative weight-synthesis fallback (Text2Weight/D2NWG-style) for rare/data-poor tasks, and closing the loop from `flywheel.HardCaseStore` back into retraining.

## Program context

Coordinate via `aixavier/FABLE_QUESTIONS.md` (claims/actions/Q&A,
append-only - append, don't rewrite history). Sister products integrate via
artifacts only, never source imports: D1 (certified selective perception)
supplies the uncertainty-head contract this factory's calibration stage
targets; A1 (fixed-camera foundation model) is a future backbone source for
`models.py`'s registry; C3 (active-learning flywheel) is the intended
consumer of `flywheel.HardCaseStore`'s captured hard cases.

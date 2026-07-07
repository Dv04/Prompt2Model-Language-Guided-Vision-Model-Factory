# Prompt2Model

`Prompt2Model` is a week-4 integrated prototype for the course project "Prompt2Model: Language-Guided Vision Model Factory".

What is implemented:

- Dev Sanghvi track: typed pipeline schema, deterministic prompt parser, augmentation mapping, CLIP-ready label resolver with lexical fallback
- Venkata track: classification and COCO-style detection dataset loaders, lightweight model registry, training loops, augmentation injection
- Madhuvani track: metric harness, ONNX export with metadata injection, ONNX verification, markdown evaluation report generation
- B1 factory-compiler track (all opt-in, offline-first): an LLM planner front end, a distill → quantize → accuracy-floor compression gate, pluggable deployment targets, calibration + conformal abstain, and a flywheel hard-case store

What is validated:

- Classification path runs end-to-end on synthetic data: prompt -> config -> training -> metrics -> ONNX export -> report
- Detection path is integrated through the week-4 milestone: prompt -> config -> COCO loader -> detector training/eval smoke test

## Repo Layout

- `src/prompt2model/`: package source
- `tests/`: smoke tests and unit tests
- `latex/`: proposal PDF and LaTeX source
- `docs/`: project package PDF and generated markdown notes

## Environment

The quickest reproducible setup in this repo is a local virtualenv that can still see the machine's installed PyTorch build:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
```

Note: `pyproject.toml` now pins `ml_dtypes>=0.4` explicitly — `onnxruntime.quantization`
(the INT8 PTQ path used by the compression stage) imports it but `onnxruntime`
itself does not declare it as a dependency.

## Quick Start

Generate toy datasets:

```bash
.venv/bin/python -m prompt2model.cli generate-toy-data --task all --output-dir output/toy_data
```

Run the built-in smoke demo:

```bash
.venv/bin/python -m prompt2model.cli smoke-test --output-dir output/smoke
```

Run a manual classification job:

```bash
.venv/bin/python -m prompt2model.cli run \
  --prompt "Classify red square, blue circle, and green triangle images under low light and prioritize speed." \
  --task classification \
  --dataset-root output/toy_data/classification \
  --dataset-format imagefolder \
  --output-dir output/manual_classification
```

Run tests:

```bash
.venv/bin/python -m pytest
```

Everything below this point is opt-in: with no flags and no `P2M_LLM_ENDPOINT`
set, `run` behaves exactly as above — regex parsing, no compression, ONNX
Runtime output, no calibration metadata beyond what already existed.

## Constraints the planner/CLI can set

`ModelConstraints` (`config.py`) carries, in addition to `priority` /
`target_latency_ms` / `budget_minutes`:

- `recall_bias` (0-1) — recall-vs-precision preference ("never miss one" -> high; "no false alarms" -> low)
- `power_budget_w` — device power envelope in watts; at/below 5 W the speed tier is forced
- `accuracy_floor` (0-1) — absolute minimum acceptable accuracy; feeds the compression gate (see below)
- `max_parameters_millions` — hard cap on model size

`models.recommend_model()` applies these during model selection: `priority`
picks the starting tier (speed/balanced/accuracy), and `power_budget_w` /
`target_latency_ms` / `max_parameters_millions` can only push the choice
**down** to a smaller/faster tier — a stated constraint always wins over a
stated preference, never the other way around.

## LLM planner (prompt -> typed spec)

The regex parser (`parsing.parse_prompt`) stays the default and is
dependency-free, but it can't read indirect phrasing ("never miss a single
one" as a recall preference, "runs at 2 watts" as a power budget). `planner.py`
puts an LLM in front of it, talking to **any OpenAI-compatible
`/v1/chat/completions` endpoint** (Ollama, vLLM, llama.cpp server, OpenAI):

```
prompt ──► LLM ──► strict JSON ──► PlannerOutput (validated) ──► overlay onto
                                          the regex-parsed PipelineConfig
```

- **Offline-first**: no endpoint configured -> the regex parser runs alone,
  byte-for-byte the previous behaviour. `mode="auto"` (the CLI default) uses
  the LLM when configured and falls back to regex on any planner failure —
  a config is always produced. `mode="llm"` makes a configured endpoint
  mandatory and raises `PlannerError` if it fails; `mode="regex"` skips the
  LLM entirely.
- **Overlay, never replace**: the regex parse always produces a complete
  config; the planner only overwrites fields it actually extracted
  (non-null). A half-answer from a small model can't hole the config. An
  explicit `--task` always outranks whatever the LLM infers.
- **One repair round-trip**: invalid/non-JSON output gets one corrective
  follow-up message before the planner gives up (`PlannerError` in `llm`
  mode, silent fallback in `auto`).

Env vars (or constructor args): `P2M_LLM_ENDPOINT`, `P2M_LLM_MODEL`,
`P2M_LLM_TIMEOUT` (seconds, default 60).

CLI flags: `--planner {auto,llm,regex}` (default `auto`), `--llm-endpoint`,
`--llm-model` — these override the env vars for a single run.

Example against a local Ollama endpoint:

```bash
export P2M_LLM_ENDPOINT=http://localhost:11434
export P2M_LLM_MODEL=qwen3.5

.venv/bin/python -m prompt2model.cli run \
  --prompt "I need to spot forklifts without a driver's helmet, we can never miss one, runs at 2 watts on the loading dock camera, quantized int8" \
  --task detection \
  --dataset-root output/toy_data/detection \
  --dataset-format coco \
  --annotation-path output/toy_data/detection/annotations.json \
  --planner llm \
  --output-dir output/manual_planner
```

`--llm-endpoint http://localhost:11434 --llm-model qwen3.5` works the same way
without exporting the env vars first.

## Compression: distill -> quantize -> accuracy-floor gate

`compression.py`'s promise is not "a smaller model" but "a smaller model OR a
refusal". Enable it with `--quantize` / `--distill`, or let the prompt/planner
turn it on (keywords like "quantized int8" / "distilled", or `PlannerOutput.quantize`
/ `.distill`).

- **Distillation** (`train_distilled_classification`, classification only):
  soft-target KL (temperature-scaled) + hard-label CE against a teacher
  trained from the registry's accuracy tier
  (`CompressionConfig.distillation_teacher`, default temperature `4.0`,
  alpha `0.7`).
- **Quantization** (`quantize_onnx`): INT8 post-training quantization of the
  exported ONNX file via `onnxruntime.quantization`. `dynamic` (default)
  quantizes weights only and needs no calibration data; `static` additionally
  calibrates activations from the validation loader
  (`CompressionConfig.quantization_mode`).
- **The gate** (`decide_gate` / `apply_compression`): the compressed artifact
  is evaluated in ONNX Runtime — the SAME runtime it ships in — and compared
  against `max(accuracy_floor_relative * baseline_accuracy, constraints.accuracy_floor)`
  (default relative floor `0.98`, i.e. at most a 2% accuracy hit). If it
  can't hold that floor, **the factory refuses to ship the compressed
  artifact and keeps the uncompressed one** — this refusal is the safety
  contract, not a bug to work around.

`PipelineResult` carries `compression` (the gate report: baseline/compressed
accuracy, floor, `passed`, sizes, `reason`) and `compressed_onnx_path` (set
only when the gate passed).

```bash
.venv/bin/python -m prompt2model.cli run \
  --prompt "Classify red square, blue circle, and green triangle images, keep at least 90% accuracy" \
  --task classification \
  --dataset-root output/toy_data/classification \
  --dataset-format imagefolder \
  --quantize \
  --output-dir output/manual_quantized
```

A `"REFUSED: compressed artifact fell below the accuracy floor — shipping
uncompressed"` reason in the report means exactly that: check `compression.passed`
in the JSON output before assuming the INT8 file was actually shipped.

## Deployment targets

`targets.py`: the factory compiles to a **target**, not a box. ONNX Runtime
is the universal backend every artifact runs on unmodified; accelerator
backends (TensorRT first) are one registry entry each.

```bash
.venv/bin/python -m prompt2model.cli run \
  --prompt "Classify red square, blue circle, and green triangle images" \
  --task classification \
  --dataset-root output/toy_data/classification \
  --dataset-format imagefolder \
  --target jetson \
  --output-dir output/manual_trt
```

- `--target onnxruntime` / `cpu` / `ort` / `host` (default): verifies the
  ONNX file loads in `onnxruntime.InferenceSession` and ships it as-is.
- `--target tensorrt` / `trt` / `jetson` / `orin`: builds the TensorRT engine
  locally via `trtexec` when it's found on `PATH` (checking the JetPack
  fallback `/usr/src/tensorrt/bin/trtexec` too); when it isn't, emits a
  reproducible `build_tensorrt.sh` into the run directory to build **on the
  target device** instead — TensorRT engines are tied to a specific GPU +
  TensorRT version, so building on-device is the normal path, not a fallback
  of last resort.
- Unknown target names fall back to `onnxruntime` (with a warning log), never
  an error — the factory always produces something deployable.

`PipelineResult.deployment` carries the target name, runtime, artifact path,
`built` flag, and `recipe_path` when a build script was emitted instead of a
local build.

## Calibration + abstain

Every classification artifact ships knowing when it doesn't know.
`calibration.py`'s `calibrate_classification` fits, on the validation split,
and embeds both values in the ONNX file's custom metadata so they travel with
the artifact:

- **temperature** — a single scalar dividing the logits before softmax,
  fitted by log-spaced grid search over `[0.05, 10]` to minimize validation
  NLL. `ece_before` / `ece_after` (Expected Calibration Error) are recorded
  in the metadata/metrics so the effect is auditable.
- **conformal abstain threshold** — the split-conformal `(1 - alpha)`
  quantile of validation nonconformity (`1 - calibrated max-probability`),
  `alpha=0.1` by default. At inference, nonconformity above the threshold
  means ABSTAIN.

`EdgeModel` (`edge_inference.py`) reads this metadata at load time and
applies it in `run_inference`: it temperature-scales the logits and adds
`calibrated` and `abstained` fields to the classification result. Calibration
failures never kill a run — `calibrate_classification` returns
`{"calibrated": False, ...}` on any error, and artifacts exported before this
stage existed (no `calibration` metadata block) simply load with
`calibrated=False`, `abstained=False` — old artifacts keep working.

## Flywheel (hard-case capture)

`flywheel.HardCaseStore` is the capture half of the retrain loop: labeling
and active-learning curation are a separate product and deliberately out of
scope here — "nothing captured, nothing to learn from" is the whole job of
this module.

Pass a store into inference to capture automatically:

```python
from prompt2model.edge_inference import EdgeModel
from prompt2model.flywheel import HardCaseStore

model = EdgeModel("output/manual_run/model.onnx")
store = HardCaseStore("output/hard_cases", max_items=1000, capture_below=0.5)
result = model.run_inference("frame.jpg", hard_case_store=store)
```

A frame is captured when the model abstained (conformal gate) OR its
confidence fell below `capture_below` (default `0.5`) — useful even for
uncalibrated artifacts. The store is bounded (`max_items`, drop-newest) so a
confused model in the field can't fill a disk.

Inspect or export the store from the CLI:

```bash
.venv/bin/python -m prompt2model.cli flywheel --store output/hard_cases --action status

.venv/bin/python -m prompt2model.cli flywheel --store output/hard_cases \
  --action export --output-dir output/flywheel_export --pseudo-label
```

`--action status` prints `HardCaseStore.summary()` (count, abstained count,
by-prediction breakdown). `--action export` materializes the pool as an
imagefolder via `export_imagefolder`: by default everything lands under
`unlabeled/` (hard cases need human or active-learning labels); `--pseudo-label`
buckets by the model's own prediction instead — cheap, biased, useful only
for semi-supervised recipes, with provenance kept in the manifest either way.

## Notes

- CLIP-based label resolution is implemented behind a lazy loader. It falls back to lexical matching if the CLIP model is unavailable.
- The default augmentation backend is torchvision-native for stability. The module boundary is ready for an Albumentations adapter later.
- Detection ONNX export is present as a best-effort wrapper, but the fully validated export path in this milestone is classification.
- Compression, deployment-target compilation, calibration, and flywheel capture are documented above for classification; detection support follows the same interfaces where applicable but is not yet the validated path for those stages.

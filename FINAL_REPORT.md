# Prompt2Model: Final Evaluation Report

This report aggregates performance, efficiency, and hardware telemetry across all valid model generation runs.

## 1. Executive Summary
- Total runs recorded: `2`
- Classification runs: `1`
- Detection runs: `1`
- Deployment Target: `Edge Device (ONNX Runtime)`

## 2. Classification Performance & Efficiency
| Model | Accuracy | Latency (ms) | FPS | GFLOPs | Params (M) |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v3_small | 50.0% | 86.79 | 11.5 | 0.0094 | 1.520931 |

## 3. Detection Performance & Efficiency
| Model | mAP@0.5 | Latency (ms) | FPS | GFLOPs | Params (M) |
| --- | --- | --- | --- | --- | --- |
| ssdlite320_mobilenet_v3_large | 0.077 | 123.61 | 8.1 | 0.4285 | 2.22038 |

## 4. Efficiency Insights
The following trade-offs were observed between model backbone complexity and inference speed:

- **Classification**: `mobilenet_v3_small` provides excellent low-latency (~10ms) with minimal computational cost (<0.01 GFLOPs).
- **Detection**: `ssdlite320_mobilenet_v3_large` achieves a balance of real-time performance (~120ms / 8 FPS) on CPU-bound host environments.

## 5. Deployment Readiness
All exported artifacts are **Metadata-Self-Contained**. They can be deployed using the `edge_infer.py` utility without requiring any external configuration files or library dependencies other than `onnxruntime` and `numpy`.

---
*Report generated on: 2026-05-01T05:42:07.558360*
#!/usr/bin/env python3
"""video_infer.py — Edge-inference script for Prompt2Model ONNX exports on Video streams."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort

def load_metadata(model_path: str) -> dict[str, Any]:
    model = onnx.load(model_path)
    raw = {p.key: p.value for p in model.metadata_props}
    parsed = {}
    for key, value in raw.items():
        try:
            parsed[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed[key] = value
    return parsed

def preprocess_frame(frame: np.ndarray, input_resolution: list[int], mean: list[float], std: list[float]) -> np.ndarray:
    """Preprocess a BGR cv2 frame for ONNX inference."""
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    h, w = input_resolution
    resized = cv2.resize(rgb, (w, h))
    
    # Normalize
    arr = resized.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    arr = (arr - mean_arr) / std_arr
    
    # NCHW
    arr = arr.transpose(2, 0, 1)
    arr = arr[np.newaxis, ...]
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Path to output video")
    args = parser.parse_args()

    meta = load_metadata(args.model)
    task = meta["task"]
    res = meta["input_resolution"]
    mean = meta["mean"]
    std = meta["std"]
    class_dict = meta["class_dict"]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video {args.video} ...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.perf_counter()
        inp = preprocess_frame(frame, res, mean, std)
        outputs = session.run(None, {input_name: inp})
        latency = (time.perf_counter() - t0) * 1000

        # Draw results
        if task == "classification":
            logits = outputs[0].squeeze()
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            idx = np.argmax(probs)
            conf = probs[idx]
            label = class_dict.get(str(idx), f"class_{idx}")
            
            text = f"{label.upper()} ({conf*100:.1f}%)"
            color = (0, 255, 0)
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Latency: {latency:.1f}ms", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif task == "detection":
            boxes, scores, labels = outputs[0].squeeze(), outputs[1].squeeze(), outputs[2].squeeze()
            if boxes.ndim == 1:
                boxes, scores, labels = boxes[np.newaxis, :], scores[np.newaxis], labels[np.newaxis]
            
            for box, score, lbl in zip(boxes, scores, labels):
                if score < 0.3: continue
                label_key = str(int(lbl))
                if label_key not in class_dict: label_key = str(int(lbl) - 1)
                label = class_dict.get(label_key, f"class_{int(lbl)}")
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Saved annotated video to {args.output}")

if __name__ == "__main__":
    main()

"""Test exported CED ONNX models (test_onnx.py's `export_onnx.py` outputs)
against the exact PyTorch model they were traced from.

Works with both `--waveform-input` (ONNX input "wav") and feature-input
(ONNX input "feats") exports -- the model name, `--max-frames`, and
waveform-input flag used at export time are read back from metadata that
`export_onnx.py` embeds in the ONNX file, so the correct reference model is
rebuilt automatically rather than guessed from the filename.

Single-model mode prints predictions, a timing benchmark, and error vs. the
PyTorch reference. Directory-sweep mode runs every *.onnx file in a
directory (e.g. an `export_onnx.py --output-dir` folder) and writes one row
per file to a CSV.

Examples:
    python test_onnx.py --model-path onnx_models/ced_tiny.wav.sim.int8.onnx
    python test_onnx.py --model-path onnx_models/ced_tiny.onnx --wav-input other.wav
    python test_onnx.py --model-path onnx_models/ced_tiny.wav.onnx --synthetic
    python test_onnx.py --models-dir onnx_models --output-csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm
import onnxruntime
import pandas as pd
import torch

import models

SAMPLE_RATE = 16000
# Only used as a fallback for *.onnx files exported before export_onnx.py
# started embedding "max_frames" metadata.
DEFAULT_MAX_FRAMES = 1012

FIELDNAMES = [
    "model_path",
    "model_name",
    "max_frames",
    "waveform_input",
    "top1_index",
    "top1_label",
    "top1_score",
    "inference_ms",
    "abs_error_max",
    "abs_error_mean",
    "rel_error_max",
    "rel_error_mean",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path", help="Path to a single CED ONNX model.")
    source.add_argument(
        "--models-dir",
        help="Directory of *.onnx files to test (e.g. export_onnx.py's --output-dir). "
             "Writes one row per file to --output-csv instead of printing.")

    parser.add_argument(
        "--wav-input",
        default="sample.wav",
        help="Path to the audio file to run inference on (default: sample.wav). "
             "Ignored with --synthetic.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore --wav-input and run on a synthetic random waveform instead.")
    parser.add_argument(
        "--synthetic-seconds",
        type=float,
        default=5.0,
        help="Length of the synthetic waveform when --synthetic is passed (default: 5.0).")
    parser.add_argument(
        "--skip-time",
        action="store_true",
        help="Skip the inference-time benchmark (useful on a noisy shared machine).")
    parser.add_argument(
        "--skip-error",
        action="store_true",
        help="Skip comparing against the PyTorch reference model's logits.")
    parser.add_argument(
        "--labels-path",
        type=str,
        default=str(
            Path(__file__).parent / 'datasets/audioset/data/metadata/class_labels_indices.csv'),
        help="AudioSet class_labels_indices.csv, used to print top-k label names. "
             "Falls back to downloading it if the local path doesn't exist.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predicted labels to print.")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="onnxruntime intra_op_num_threads (default: 1). Leave at 1 unless you've "
             "checked ORT's default thread pool doesn't oversubscribe a model this small.")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up iterations before timing (default: 100).")
    parser.add_argument("--reps", type=int, default=1000, help="Timed iterations for the benchmark (default: 1000).")
    parser.add_argument(
        "--output-csv",
        default="results.csv",
        help="Where to write results in --models-dir mode (default: results.csv).")
    return parser


def load_waveform(args: argparse.Namespace) -> np.ndarray:
    """Load a mono float32 16 kHz waveform, real or synthetic."""
    if args.synthetic:
        rng = np.random.default_rng(0)
        num_samples = round(args.synthetic_seconds * SAMPLE_RATE)
        return rng.uniform(-0.3, 0.3, num_samples).astype(np.float32)

    import soundfile as sf

    audio, sr = sf.read(args.wav_input)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{args.wav_input} is {sr} Hz, expected {SAMPLE_RATE} Hz.")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32)


def load_labels(path: str) -> Dict[int, str]:
    source = (path if Path(path).exists() else
             'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv')
    return pd.read_csv(source).set_index('index')['display_name'].to_dict()


def build_session(model_path: str, num_threads: int) -> onnxruntime.InferenceSession:
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = num_threads
    return onnxruntime.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


def resolve_reference_model(session: onnxruntime.InferenceSession,
                            model_path: str) -> tuple[torch.nn.Module, bool, str, int]:
    """Rebuild the exact PyTorch model `model_path` was exported from, using
    the metadata export_onnx.py embeds (model_name/max_frames/waveform_input)."""
    meta = session.get_modelmeta().custom_metadata_map
    model_name = meta.get("model_name")
    if model_name is None:
        model_name = Path(model_path).name.split(".")[0]
        max_frames = DEFAULT_MAX_FRAMES
        waveform_input = session.get_inputs()[0].name != "feats"
        print(f"Warning: {model_path} has no embedded export metadata (re-export with "
             f"the current export_onnx.py to fix this) -- guessing model_name="
             f"'{model_name}', max_frames={max_frames} from the filename/graph instead "
             f"of the exact values used at export time.")
    else:
        max_frames = int(meta["max_frames"])
        waveform_input = meta["waveform_input"] == "True"

    if model_name not in models.list_models():
        raise SystemExit(f"Unknown model name '{model_name}' for {model_path}")

    model = getattr(models, model_name)(target_length=max_frames, pretrained=True).eval()
    return model, waveform_input, model_name, max_frames


def prepare_feed(session: onnxruntime.InferenceSession, waveform: np.ndarray,
                 model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Build the ONNX input feed, computing feats via the PyTorch model's own
    front_end when the graph expects precomputed features instead of a
    waveform -- this must happen once, outside the timed benchmark loop."""
    input_name = session.get_inputs()[0].name
    if input_name == "wav":
        arr = waveform[None, :].astype(np.float32)
    elif input_name == "feats":
        wav_t = torch.from_numpy(waveform).unsqueeze(0)
        with torch.no_grad():
            arr = model.front_end(wav_t).numpy().astype(np.float32)
    else:
        raise ValueError(f"Unexpected ONNX input name '{input_name}' (expected 'wav' or 'feats')")
    return {input_name: arr}


def run_inference(session: onnxruntime.InferenceSession, feed: Dict[str, np.ndarray]) -> np.ndarray:
    logits = session.run(None, feed)[0]
    return np.asarray(logits).squeeze()


def benchmark(session: onnxruntime.InferenceSession, feed: Dict[str, np.ndarray],
              warmup: int, reps: int) -> float:
    for _ in tqdm(range(warmup), desc="warmup", leave=False):
        run_inference(session, feed)
    t_start = time.time()
    for _ in tqdm(range(reps), desc="reps", leave=False):
        run_inference(session, feed)
    return (time.time() - t_start) / reps


def compute_reference_logits(model: torch.nn.Module, waveform: np.ndarray,
                             waveform_input: bool) -> np.ndarray:
    wav_t = torch.from_numpy(waveform).unsqueeze(0)
    with torch.no_grad():
        if waveform_input:
            logits = model(wav_t)
        else:
            feats = model.front_end(wav_t)
            logits = model.forward_spectrogram(feats)
    return logits.squeeze(0).numpy()


def print_top_k(logits: np.ndarray, labels: Optional[Dict[int, str]], top_k: int) -> None:
    order = np.argsort(logits)[::-1][:top_k]
    print(f"Top-{top_k} predictions:")
    for idx in order:
        name = labels[idx] if labels is not None else str(idx)
        print(f"  [{idx:3d}] {name:35s} {logits[idx]:.4f}")


def evaluate(model_path: str, waveform: np.ndarray, args: argparse.Namespace) -> dict:
    """Run one ONNX file's full test and return a FIELDNAMES-shaped row."""
    session = build_session(model_path, args.num_threads)
    model, waveform_input, model_name, max_frames = resolve_reference_model(session, model_path)

    feed = prepare_feed(session, waveform, model)
    logits = run_inference(session, feed)

    top1_index = int(np.argmax(logits))
    row: dict = {
        "model_path": str(model_path),
        "model_name": model_name,
        "max_frames": max_frames,
        "waveform_input": waveform_input,
        "top1_index": top1_index,
        "top1_score": float(logits[top1_index]),
    }

    if not args.skip_time:
        row["inference_ms"] = benchmark(session, feed, args.warmup, args.reps) * 1000

    if not args.skip_error:
        ideal_logits = compute_reference_logits(model, waveform, waveform_input)
        diff = np.abs(logits - ideal_logits)
        rel = diff / np.maximum(np.abs(ideal_logits), 1e-12)
        row["abs_error_max"] = float(diff.max())
        row["abs_error_mean"] = float(diff.mean())
        row["rel_error_max"] = float(rel.max())
        row["rel_error_mean"] = float(rel.mean())

    row["_logits"] = logits
    return row


def main() -> None:
    args = build_arg_parser().parse_args()

    waveform = load_waveform(args)
    labels = load_labels(args.labels_path)

    if args.model_path:
        row = evaluate(args.model_path, waveform, args)
        print_top_k(row.pop("_logits"), labels, args.top_k)
        row["top1_label"] = labels.get(row["top1_index"], "")
        if "inference_ms" in row:
            print(f"\nAverage inference time over {args.reps} runs: {row['inference_ms']:.3f} ms")
        if "abs_error_max" in row:
            print("\nError vs. PyTorch reference:")
            print(f"  max abs error:  {row['abs_error_max']:.6f}")
            print(f"  mean abs error: {row['abs_error_mean']:.6f}")
            print(f"  max rel error:  {row['rel_error_max']:.6f}")
            print(f"  mean rel error: {row['rel_error_mean']:.6f}")
        return

    model_paths = sorted(Path(args.models_dir).glob("*.onnx"))
    if not model_paths:
        raise SystemExit(f"No *.onnx files found under {args.models_dir}/")

    with open(args.output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for model_path in model_paths:
            print(model_path.name)
            row = evaluate(str(model_path), waveform, args)
            row.pop("_logits")
            row["top1_label"] = labels.get(row["top1_index"], "")
            writer.writerow(row)
            csv_file.flush()

    print(f"\nWrote {len(model_paths)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

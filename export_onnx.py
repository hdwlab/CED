import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

import models

DEVICE = torch.device('cpu')


def simplify_onnx(input_path: str, output_path: str):
    """Run onnxsim on `input_path`, save to `output_path`, and print a diff
    of node/initializer counts and per-op-type counts -- only if anything
    actually decreased, to keep the log quiet otherwise."""
    orig = onnx.load(input_path)
    # NOTE: perform_optimization=True seems to slow down inference. 
    # Might be recommended to set to False.
    sim_model, check = simplify(orig, perform_optimization=True)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(sim_model, output_path)

    orig_nodes = list(orig.graph.node)
    sim_nodes = list(sim_model.graph.node)
    n_orig_init = len(orig.graph.initializer)
    n_sim_init = len(sim_model.graph.initializer)

    if len(sim_nodes) >= len(orig_nodes) and n_sim_init >= n_orig_init:
        print(f"onnxsim: no simplification for {input_path}")
        return

    print(f"onnxsim: {input_path} -> {output_path}")
    print(f"  Nodes: {len(orig_nodes)} -> {len(sim_nodes)}")
    print(f"  Initializers: {n_orig_init} -> {n_sim_init}")

    orig_ops = Counter(n.op_type for n in orig_nodes)
    sim_ops = Counter(n.op_type for n in sim_nodes)
    changed_ops = {
        op
        for op in set(orig_ops) | set(sim_ops)
        if orig_ops.get(op, 0) != sim_ops.get(op, 0)
    }
    if changed_ops:
        print("  Ops:")
        for op in sorted(changed_ops):
            print(f"    {op}: {orig_ops.get(op, 0)} -> {sim_ops.get(op, 0)}")


def find_matmul_node_name(model: onnx.ModelProto,
                          initializer_substr: str) -> Optional[str]:
    """Find a MatMul node's name by an initializer name it takes as input.
    Node names can change across onnxsim, so this must be re-resolved on
    the simplified model rather than assumed from the original export."""
    for node in model.graph.node:
        if node.op_type == 'MatMul' and any(initializer_substr in inp
                                            for inp in node.input):
            return node.name
    return None


class LowRankLinear(nn.Module):
    """SVD-based low-rank approximation of an `nn.Linear`: W ~= up @ down.
    Exports to ONNX as two ordinary Gemm/MatMul ops, so it needs no special
    handling anywhere else in the export/simplify/quantize pipeline."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.up(self.down(x))

    @classmethod
    @torch.no_grad()
    def from_linear(cls,
                    linear: nn.Linear,
                    min_energy: float,
                    rank_multiples: int = 8):
        valid_ranks = tuple(
            range(rank_multiples,
                  min(linear.in_features, linear.out_features) + 1,
                  rank_multiples))
        orig_ops = linear.in_features * linear.out_features

        # W = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(linear.weight, full_matrices=False)

        energy = S**2
        energy_ratio = energy / energy.sum()
        for valid_rank in sorted(valid_ranks, reverse=False):
            rank_energy_ratio = energy_ratio[:valid_rank].sum()
            ops = linear.in_features * valid_rank + valid_rank * linear.out_features
            if (rank_energy_ratio >= min_energy) and (ops < orig_ops):
                rank = valid_rank
                break
        else:
            return linear

        layer = cls(
            linear.in_features,
            linear.out_features,
            rank,
            bias=linear.bias is not None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        # W ~= (U_r @ diag(S_r)) @ Vh_r
        layer.down.weight.copy_(Vh[:rank])
        layer.up.weight.copy_(U[:, :rank] * S[:rank])

        if linear.bias is not None:
            layer.up.bias.copy_(linear.bias)

        return layer


def replace_module(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def decompose_linears(model: nn.Module, min_energy: float,
                      rank_multiples: int):
    """In-place: replace every `nn.Linear` in `model` whose SVD rank
    (gated by `min_energy`) needs fewer FLOPs with a `LowRankLinear`."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            lora_module = LowRankLinear.from_linear(
                module, min_energy=min_energy, rank_multiples=rank_multiples)
            if lora_module is not module:
                print(f"  {name}: Linear({module.in_features} -> "
                     f"{module.out_features}) -> LowRankLinear"
                     f"({lora_module.in_features} -> {lora_module.rank} -> "
                     f"{lora_module.out_features})")
                replace_module(model, name, lora_module)


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class OnnxFrontEnd(nn.Module):
    """Re-implements `FrontEnd` (MelSpectrogram + AmplitudeToDB) with ops that
    survive ONNX export: torch's ONNX `STFT` symbolic only supports
    `return_complex=False` (a real tensor with a trailing [real, imag] axis),
    whereas torchaudio's `MelSpectrogram` always calls `torch.stft` with
    `return_complex=True` internally, which fails to export. This mirrors the
    approach used for the reference ONNX export at
    https://huggingface.co/mispeech/ced-tiny/blob/main/model.onnx (STFT ->
    ReduceL2 -> Pow -> mel filterbank matmul -> Clip -> Log).

    Note: unlike `FrontEnd`, this omits `AmplitudeToDB`'s top_db=120
    relative-to-clip-max clamp (the reference HF export omits it too), so
    output can differ slightly from `FrontEnd` on very high-dynamic-range
    clips.
    """

    def __init__(self, front_end):
        super().__init__()
        self.n_fft = front_end.n_fft
        self.hop_size = front_end.hop_size
        self.win_size = front_end.win_size
        self.register_buffer('window', front_end[0].spectrogram.window.clone())
        self.register_buffer('mel_fb', front_end[0].mel_scale.fb.clone())
        self.power = front_end[0].spectrogram.power
        self.amin = front_end[1].amin
        self.multiplier = front_end[1].multiplier

    def forward(self, wav):
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
            pad_mode='reflect',
            onesided=True,
            return_complex=False,
        )  # [B, freq, T, 2]
        mag = torch.linalg.vector_norm(spec, dim=-1)  # [B, freq, T]
        power = mag**self.power
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
        mel_db = self.multiplier * torch.log10(torch.clamp(mel, min=self.amin))
        return mel_db


class OnnxExportModel(nn.Module):
    def __init__(self, model, waveform_input: bool = False):
        super(OnnxExportModel, self).__init__()
        self.model = model
        self.waveform_input = waveform_input
        if waveform_input:
            self.front_end = OnnxFrontEnd(model.front_end)

    def forward(self, x):
        if self.waveform_input:
            mel = self.front_end(x)
            return self.model.forward_spectrogram(mel)
        # x = x.permute(0,2,1)
        # If necessary, transpose can be performed here, switching the feature dimensions with the time dimensions.
        # If transposed, the following dynamic axes should also be interchanged.
        feat = self.model.forward_spectrogram(x)
        return feat


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        metavar=
        f"Public Checkpoint [{','.join(models.list_models())}] or Experiment Path",
        nargs='?',
        choices=models.list_models(),
        default='ced_mini')

    parser.add_argument(
        '--max-frames',
        type=int,
        default=1012,
        help="Max number of frames the model can process."
        )
    parser.add_argument(
        '--waveform-input',
        action='store_true',
        help="Export front_end (STFT/mel/log) inside the ONNX graph, so the "
             "model takes raw waveform instead of precomputed feats. "
             "Requires opset >= 17 (torch.stft ONNX support)."
        )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help="Directory to store the exported ONNX models in."
        )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help="Also run onnxsim on the exported model and produce "
             "additional *.sim.onnx / *.sim.int8.onnx files."
        )
    parser.add_argument(
        '--decompose',
        action='store_true',
        help="Replace nn.Linear layers with an SVD low-rank approximation "
             "(down-projection + up-projection) before export, wherever "
             "the rank needed to keep --decompose-min-energy also reduces "
             "FLOPs. Produces an additional *.lora.onnx file."
        )
    parser.add_argument(
        '--decompose-min-energy',
        type=float,
        default=0.95,
        help="Minimum fraction of singular-value energy a layer's low-rank "
             "approximation must retain to be used."
        )
    parser.add_argument(
        '--decompose-rank-multiples',
        type=int,
        default=8,
        help="Candidate ranks are checked in multiples of this value."
        )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    model = getattr(models, args.model)(target_length=args.max_frames,
                                        pretrained=True)
    if args.waveform_input:
        # STFT with center=True yields 1 + num_samples // hop_size frames, so
        # use (max_frames - 1) * hop_size samples to trace exactly max_frames
        # frames -- matching the feature-input dummy exactly, and avoiding
        # tracing the model's long-sequence chunk-splitting branch (which
        # bakes a fixed-size Split node into the graph for any other length).
        num_samples = (args.max_frames - 1) * model.hop_size
        dummy_input = torch.ones(1, num_samples)
        input_names = ['wav']
        dynamic_axes = {
            'wav': {
                0: 'batch_size',
                1: 'num_samples'
            },
            'prob': {
                0: 'batch_size'
            }
        }
        opset_version = 17
    else:
        dummy_input = torch.ones(1, model.n_mels, args.max_frames)
        input_names = ['feats']
        dynamic_axes = {
            'feats': {
                0: 'batch_size',
                2: 'time_dim'
            },
            'prob': {
                0: 'batch_size'
            }
        }
        opset_version = 12
    model = model.to(DEVICE).eval()

    if args.decompose:
        print("Decomposing Linear layers")
        decompose_linears(model,
                          min_energy=args.decompose_min_energy,
                          rank_multiples=args.decompose_rank_multiples)

    model = OnnxExportModel(model, waveform_input=args.waveform_input)
    out = model(dummy_input)
    print(f"Model Output is {out.shape}")

    output_model = str(output_dir / (args.model +
                      ('.wav' if args.waveform_input else '') +
                      ('.lora' if args.decompose else '') + '.onnx'))
    torch.onnx.export(model,
                      dummy_input,
                      output_model,
                      do_constant_folding=True,
                      verbose=False,
                      opset_version=opset_version,
                      dynamo=False,
                      input_names=input_names,
                      output_names=['prob'],
                      dynamic_axes=dynamic_axes)

    meta_data = {
        "model_type": "CED",
        "version": "1.0",
        "model_author": "RicherMans",
        "url": "https://github.com/RicherMans/CED",
        # Lets a consumer (e.g. test_onnx.py) rebuild the exact PyTorch
        # model this graph was traced from, rather than guessing from the
        # filename.
        "model_name": args.model,
        "max_frames": str(args.max_frames),
        "waveform_input": str(args.waveform_input),
    }
    add_meta_data(filename=output_model, meta_data=meta_data)

    if args.simplify:
        print("Simplifying ONNX graph")
        sim_model = output_model.replace('.onnx', '.sim.onnx')
        simplify_onnx(output_model, sim_model)

    print("Generate int8 quantization models")

    # The front_end's mel-filterbank MatMul runs on pre-log power-spectrogram
    # values (huge dynamic range -- log-compression is the very next step),
    # so int8-quantizing it destroys accuracy. Exclude it; everything
    # downstream operates on already-log-compressed/normalized activations.
    # Node names can differ between the raw export and the simplified graph,
    # so this is re-resolved separately for each.
    def quantize(src_model: str, dst_model: str):
        nodes_to_exclude = None
        if args.waveform_input:
            mel_fb_matmul = find_matmul_node_name(onnx.load(src_model), 'mel_fb')
            nodes_to_exclude = [mel_fb_matmul] if mel_fb_matmul else None
        quantize_dynamic(
            model_input=src_model,
            model_output=dst_model,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
            nodes_to_exclude=nodes_to_exclude,
        )
        print(f"Results is at {dst_model}")

    quantize(output_model, output_model.replace('.onnx', '.int8.onnx'))
    if args.simplify:
        quantize(sim_model, sim_model.replace('.sim.onnx', '.sim.int8.onnx'))


if __name__ == "__main__":
    main()

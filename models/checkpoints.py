import os
import sys
import tempfile
import torch
from typing import Callable, Dict, Optional


_model_functions: Dict[str, Callable] = {
}  # mapping of model names to entrypoint fns


def register_model(fn: Callable) -> Callable:
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    _model_functions[model_name] = fn
    return fn


def _load_hf_safetensors(url: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from a HuggingFace-style `model.safetensors` URL
    and remap it onto `AudioTransformer`'s key names: HF's
    CedForAudioClassification wraps the backbone weights in `encoder.`, and
    stores `init_bn` (a plain BatchNorm2d there) flat, whereas here it is
    `init_bn.1` inside an `nn.Sequential(Rearrange, BatchNorm2d, Rearrange)`.
    """
    from safetensors.torch import load_file

    with tempfile.TemporaryDirectory() as tmp_dir:
        dst = os.path.join(tmp_dir, 'model.safetensors')
        torch.hub.download_url_to_file(url, dst, progress=True)
        raw = load_file(dst)

    remapped = {}
    for k, v in raw.items():
        if k.startswith('encoder.'):
            k = k[len('encoder.'):]
        if k.startswith('init_bn.'):
            k = 'init_bn.1.' + k[len('init_bn.'):]
        remapped[k] = v
    return remapped


def build_mdl(model_fn,
              pretrained: bool = False,
              pretrained_url: Optional[str] = None,
              **model_kwargs
              ):
    mdl = model_fn(**model_kwargs)
    if pretrained and pretrained_url is not None:
        if pretrained_url.endswith('.safetensors'):
            dump = _load_hf_safetensors(pretrained_url)
        elif 'http' in pretrained_url:
            dump = torch.hub.load_state_dict_from_url(pretrained_url,
                                                      map_location='cpu')
        else:
            dump = torch.load(pretrained_url, map_location='cpu')
        if 'model' in dump:
            dump = dump['model']
        mdl.load_state_dict(dump, strict=False)
    return mdl

def list_models():
    return _model_functions.keys()

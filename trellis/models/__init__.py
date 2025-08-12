import importlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file))

    return model

class HubMixin:
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        ignore_mismatched_sizes: bool = False,
        **kwargs,
    ):
        is_local = Path(f"{path}.json").exists() and Path(f"{path}.safetensors").exists()
        if is_local:
            config_file = f"{path}.json"
            model_file = f"{path}.safetensors"
        else:
            path_parts = path.split("/")
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            model_name = "/".join(path_parts[2:])
            config_file = hf_hub_download(repo_id, f"{model_name}.json")
            model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

        with Path(config_file).open() as f:
            config = json.load(f)

        merged_config = {**config["args"], **kwargs}
        model = cls(**merged_config)
        
        state_dict = load_file(model_file)

        if ignore_mismatched_sizes:
            model_state = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"[ignore] {k}: checkpoint {tuple(v.shape)} != model {tuple(model_state.get(k, None).shape)}")
            state_dict = filtered_state_dict

        # 加载
        model.load_state_dict(state_dict, strict=not ignore_mismatched_sizes)

        return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )

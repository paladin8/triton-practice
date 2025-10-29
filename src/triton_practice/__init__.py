import torch
import triton


__all__ = ["sanity_check"]


def sanity_check() -> None:
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Triton version:", triton.__version__)
    return

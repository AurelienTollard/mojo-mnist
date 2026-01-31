"""Modular Mojo linear_relu operator exposed directly to Python.

This module exposes just the optimized kernel - Python manages everything else.
"""

from pathlib import Path

import torch
from max.torch import CustomOpLibrary

_mojo_ops = CustomOpLibrary(Path(__file__).parent)


def linear_relu(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    batch_size = input.shape[0]
    in_features = input.shape[1]
    out_features = weight.shape[1]

    if weight.shape != (in_features, out_features):
        raise ValueError(
            f"Weight shape {weight.shape} doesn't match expected"
            f" ({in_features}, {out_features}). Note: Mojo expects"
            " (in_features, out_features), not PyTorch's (out, in)."
        )

    if bias.shape != (out_features,):
        raise ValueError(
            f"Bias shape {bias.shape} doesn't match ({out_features},)"
        )

    if output.shape != (batch_size, out_features):
        raise ValueError(
            f"Output shape {output.shape} doesn't match expected "
            f"({batch_size}, {out_features})"
        )

    # Ensure all tensors are on the same device
    device = input.device
    if (
        weight.device != device
        or bias.device != device
        or output.device != device
    ):
        raise ValueError("All tensors must be on the same device")

    kernel = _mojo_ops.linear_relu[
        {
            "batch_size": batch_size,
            "in_features": in_features,
            "out_features": out_features,
        }
    ]

    kernel(output, input, weight, bias)


def linear(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    batch_size = input.shape[0]
    in_features = input.shape[1]
    out_features = weight.shape[1]

    if weight.shape != (in_features, out_features):
        raise ValueError(
            f"Weight shape {weight.shape} doesn't match ({in_features},"
            f" {out_features})"
        )

    if bias.shape != (out_features,):
        raise ValueError(
            f"Bias shape {bias.shape} doesn't match ({out_features},)"
        )

    if output.shape != (batch_size, out_features):
        raise ValueError(
            f"Output shape {output.shape} doesn't match ({batch_size},"
            f" {out_features})"
        )

    # Ensure all tensors are on the same device
    device = input.device
    if (
        weight.device != device
        or bias.device != device
        or output.device != device
    ):
        raise ValueError("All tensors must be on the same device")

    kernel = _mojo_ops.linear[
        {
            "batch_size": batch_size,
            "in_features": in_features,
            "out_features": out_features,
        }
    ]

    kernel(output, input, weight, bias)


__all__ = ["linear_relu", "linear"]

"""Modular Mojo conv operators exposed directly to Python."""

from pathlib import Path

import torch
from max.torch import CustomOpLibrary

_mojo_ops = CustomOpLibrary(Path(__file__).parent)


def _detach_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach() if tensor.requires_grad else tensor


def conv2d_maxpool_relu(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: int = 1,
    padding: int = 1,
    pool_size: int = 2,
    pool_stride: int = 2,
    tile_size: int = 16,
) -> None:
    if input.ndim != 4:
        raise ValueError("Input must be NCHW")
    if weight.ndim != 4:
        raise ValueError("Weight must be OIHW")

    batch_size, in_channels, height, width = input.shape
    out_channels, weight_in_channels, kernel_height, kernel_width = weight.shape

    if in_channels != weight_in_channels:
        raise ValueError("Input channels do not match weight channels")

    if bias.shape != (out_channels,):
        raise ValueError("Bias shape does not match out_channels")

    conv_out_height = (height + 2 * padding - kernel_height) // stride + 1
    conv_out_width = (width + 2 * padding - kernel_width) // stride + 1
    out_height = (conv_out_height - pool_size) // pool_stride + 1
    out_width = (conv_out_width - pool_size) // pool_stride + 1

    if output.shape != (batch_size, out_channels, out_height, out_width):
        raise ValueError("Output shape does not match expected pooled conv output")

    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    device = input.device
    if weight.device != device or bias.device != device or output.device != device:
        raise ValueError("All tensors must be on the same device")

    kernel = _mojo_ops.conv2d_maxpool_relu[
        {
            "batch_size": batch_size,
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "out_channels": out_channels,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "tile_size": tile_size,
            "stride": stride,
            "padding": padding,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
        }
    ]

    kernel(
        output.detach(),
        _detach_if_needed(input),
        _detach_if_needed(weight),
        _detach_if_needed(bias),
    )


def conv2d(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: int = 1,
    padding: int = 1,
    tile_size: int = 16,
) -> None:
    if input.ndim != 4:
        raise ValueError("Input must be NCHW")
    if weight.ndim != 4:
        raise ValueError("Weight must be OIHW")

    batch_size, in_channels, height, width = input.shape
    out_channels, weight_in_channels, kernel_height, kernel_width = weight.shape

    if in_channels != weight_in_channels:
        raise ValueError("Input channels do not match weight channels")

    if bias.shape != (out_channels,):
        raise ValueError("Bias shape does not match out_channels")

    out_height = (height + 2 * padding - kernel_height) // stride + 1
    out_width = (width + 2 * padding - kernel_width) // stride + 1

    if output.shape != (batch_size, out_channels, out_height, out_width):
        raise ValueError("Output shape does not match expected conv output")

    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    device = input.device
    if weight.device != device or bias.device != device or output.device != device:
        raise ValueError("All tensors must be on the same device")

    kernel = _mojo_ops.conv2d[
        {
            "batch_size": batch_size,
            "in_channels": in_channels,
            "height": height,
            "width": width,
            "out_channels": out_channels,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "tile_size": tile_size,
            "stride": stride,
            "padding": padding,
        }
    ]

    kernel(
        output.detach(),
        _detach_if_needed(input),
        _detach_if_needed(weight),
        _detach_if_needed(bias),
    )


__all__ = ["conv2d", "conv2d_maxpool_relu"]

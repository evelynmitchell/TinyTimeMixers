"""Device management utilities."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get the appropriate device for computation.

    Args:
        preference: Device preference - "auto", "cpu", "cuda", or "mps"

    Returns:
        torch.device for computation
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    elif preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(
    tensor: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Move tensor to device.

    Args:
        tensor: Input tensor
        device: Target device (uses auto if None)

    Returns:
        Tensor on target device
    """
    if device is None:
        device = get_device("auto")
    elif isinstance(device, str):
        device = get_device(device)
    return tensor.to(device)


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary with device availability info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "default_device": str(get_device("auto")),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info

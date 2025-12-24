"""Data augmentation for time series."""

import numpy as np
import torch


def downsample(
    data: torch.Tensor,
    factor: int,
    method: str = "subsample",
) -> torch.Tensor:
    """Downsample time series.

    Args:
        data: Input tensor (..., seq_len)
        factor: Downsampling factor
        method: "subsample" (pick every nth) or "average" (average n points)

    Returns:
        Downsampled tensor
    """
    if factor <= 1:
        return data

    if method == "subsample":
        return data[..., ::factor]
    elif method == "average":
        seq_len = data.shape[-1]
        new_len = seq_len // factor
        # Truncate to multiple of factor
        data = data[..., : new_len * factor]
        # Reshape and average
        shape = list(data.shape[:-1]) + [new_len, factor]
        return data.reshape(shape).mean(dim=-1)
    else:
        raise ValueError(f"Unknown method: {method}")


def multi_resolution_augment(
    data: torch.Tensor,
    factors: list[int] | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """Generate multiple resolution versions of data.

    Used for pre-training with diverse resolutions.

    Args:
        data: Input tensor (..., seq_len)
        factors: List of downsampling factors (default: [1, 2, 4, 8])

    Returns:
        List of (downsampled_data, factor) tuples
    """
    if factors is None:
        factors = [1, 2, 4, 8]

    results = []
    for factor in factors:
        downsampled = downsample(data, factor)
        results.append((downsampled, factor))

    return results


def add_noise(
    data: torch.Tensor,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Add Gaussian noise to data.

    Args:
        data: Input tensor
        noise_std: Standard deviation of noise (relative to data std)

    Returns:
        Noisy data
    """
    data_std = data.std()
    noise = torch.randn_like(data) * noise_std * data_std
    return data + noise


def jitter(
    data: torch.Tensor,
    sigma: float = 0.03,
) -> torch.Tensor:
    """Add random jitter to time series.

    Args:
        data: Input tensor
        sigma: Jitter magnitude

    Returns:
        Jittered data
    """
    return data + torch.randn_like(data) * sigma


def scaling(
    data: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Apply random scaling to time series.

    Args:
        data: Input tensor
        sigma: Scaling magnitude

    Returns:
        Scaled data
    """
    factor = torch.randn(1, device=data.device) * sigma + 1.0
    return data * factor


def magnitude_warp(
    data: torch.Tensor,
    sigma: float = 0.2,
    knot: int = 4,
) -> torch.Tensor:
    """Apply smooth magnitude warping.

    Args:
        data: Input tensor (..., seq_len)
        sigma: Warping magnitude
        knot: Number of knots for warping curve

    Returns:
        Warped data
    """
    seq_len = data.shape[-1]

    # Generate random warp curve
    orig_steps = np.arange(seq_len)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    warp_steps = np.linspace(0, seq_len - 1, num=knot + 2)

    # Interpolate to full length
    warp_curve = np.interp(orig_steps, warp_steps, random_warps)
    warp_curve = torch.from_numpy(warp_curve).float().to(data.device)

    return data * warp_curve


def time_warp(
    data: torch.Tensor,
    sigma: float = 0.2,
    knot: int = 4,
) -> torch.Tensor:
    """Apply smooth time warping.

    Args:
        data: Input tensor (channels, seq_len)
        sigma: Warping magnitude
        knot: Number of knots for warping curve

    Returns:
        Warped data
    """
    seq_len = data.shape[-1]

    # Generate random time warp
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    warp_steps = np.cumsum(np.concatenate([[0], random_warps]))
    warp_steps = warp_steps / warp_steps[-1] * (seq_len - 1)

    # Interpolate data at new time points
    warped_steps = np.interp(
        np.linspace(0, len(warp_steps) - 1, seq_len),
        np.arange(len(warp_steps)),
        warp_steps,
    )

    # Linear interpolation of data
    result = torch.zeros_like(data)
    for i, t in enumerate(warped_steps):
        lower = int(np.floor(t))
        upper = min(lower + 1, seq_len - 1)
        weight = t - lower
        result[..., i] = (1 - weight) * data[..., lower] + weight * data[..., upper]

    return result


def window_slice(
    data: torch.Tensor,
    reduce_ratio: float = 0.9,
) -> torch.Tensor:
    """Extract a random window from the time series.

    Args:
        data: Input tensor (..., seq_len)
        reduce_ratio: Fraction of sequence to keep

    Returns:
        Sliced data
    """
    seq_len = data.shape[-1]
    target_len = int(seq_len * reduce_ratio)

    if target_len >= seq_len:
        return data

    start = np.random.randint(0, seq_len - target_len)
    return data[..., start : start + target_len]


class TimeSeriesAugmenter:
    """Combined augmenter applying multiple transformations."""

    def __init__(
        self,
        jitter_sigma: float = 0.03,
        scaling_sigma: float = 0.1,
        magnitude_warp_sigma: float = 0.2,
        time_warp_sigma: float = 0.2,
        noise_std: float = 0.01,
        apply_prob: float = 0.5,
    ):
        """Initialize TimeSeriesAugmenter.

        Args:
            jitter_sigma: Jitter magnitude
            scaling_sigma: Scaling magnitude
            magnitude_warp_sigma: Magnitude warp sigma
            time_warp_sigma: Time warp sigma
            noise_std: Noise standard deviation
            apply_prob: Probability of applying each augmentation
        """
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.time_warp_sigma = time_warp_sigma
        self.noise_std = noise_std
        self.apply_prob = apply_prob

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations.

        Args:
            data: Input tensor

        Returns:
            Augmented tensor
        """
        if np.random.random() < self.apply_prob:
            data = jitter(data, self.jitter_sigma)

        if np.random.random() < self.apply_prob:
            data = scaling(data, self.scaling_sigma)

        if np.random.random() < self.apply_prob:
            data = add_noise(data, self.noise_std)

        return data

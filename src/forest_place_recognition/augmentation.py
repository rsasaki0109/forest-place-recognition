"""Seasonal appearance augmentation for forest images.

Provides colour/brightness transforms that simulate seasonal changes
without requiring real cross-season image pairs.  Useful for data
augmentation and unit-testing the seasonal evaluation pipeline.
"""

from __future__ import annotations

import numpy as np


def summer_to_winter(image: np.ndarray) -> np.ndarray:
    """Simulate a summer-to-winter appearance change.

    Applies desaturation, a blue colour shift, and brightness reduction.

    Parameters
    ----------
    image:
        Input RGB image as ``uint8`` array of shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Transformed image (same shape and dtype).
    """
    img = image.astype(np.float32)

    # Desaturate toward grayscale
    gray = np.mean(img, axis=2, keepdims=True)
    img = img * 0.35 + gray * 0.65

    # Blue shift: boost blue channel, suppress green/red slightly
    img[:, :, 0] *= 0.85  # R
    img[:, :, 1] *= 0.90  # G
    img[:, :, 2] *= 1.15  # B

    # Brightness reduction
    img *= 0.80

    return np.clip(img, 0, 255).astype(np.uint8)


def winter_to_summer(image: np.ndarray) -> np.ndarray:
    """Simulate a winter-to-summer appearance change.

    Applies saturation boost, a green colour shift, and brightness increase.

    Parameters
    ----------
    image:
        Input RGB image as ``uint8`` array of shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Transformed image (same shape and dtype).
    """
    img = image.astype(np.float32)

    # Boost saturation (move away from gray)
    gray = np.mean(img, axis=2, keepdims=True)
    img = gray + (img - gray) * 1.5

    # Green shift
    img[:, :, 0] *= 0.95  # R
    img[:, :, 1] *= 1.15  # G
    img[:, :, 2] *= 0.85  # B

    # Brightness increase
    img *= 1.15

    return np.clip(img, 0, 255).astype(np.uint8)


def simulate_season_change(
    image: np.ndarray,
    target_season: str,
) -> np.ndarray:
    """Apply a colour/brightness transform to simulate a seasonal change.

    Parameters
    ----------
    image:
        Input RGB image as ``uint8`` array of shape ``(H, W, 3)``.
    target_season:
        Target season — ``"winter"`` or ``"summer"``.

    Returns
    -------
    np.ndarray
        Transformed image.

    Raises
    ------
    ValueError
        If *target_season* is not ``"winter"`` or ``"summer"``.
    """
    season = target_season.lower()
    if season == "winter":
        return summer_to_winter(image)
    if season == "summer":
        return winter_to_summer(image)
    raise ValueError(
        f"Unknown target season '{target_season}'. Use 'winter' or 'summer'."
    )

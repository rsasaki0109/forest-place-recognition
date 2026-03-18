"""Abstract base class for VPR backends."""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np


class VPRBackend(abc.ABC):
    """Base class that every VPR feature-extraction backend must implement."""

    @abc.abstractmethod
    def extract(self, image_path: Path) -> np.ndarray:
        """Extract a global descriptor from a single image.

        Parameters
        ----------
        image_path:
            Path to an RGB image file.

        Returns
        -------
        np.ndarray
            1-D descriptor vector.
        """

    def extract_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract descriptors for a list of images.

        The default implementation calls :meth:`extract` sequentially.
        Subclasses may override for batched GPU inference.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, D)``.
        """
        return np.stack([self.extract(p) for p in image_paths])

    @property
    @abc.abstractmethod
    def descriptor_dim(self) -> int:
        """Dimensionality of the output descriptor."""

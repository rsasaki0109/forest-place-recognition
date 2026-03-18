"""Color histogram baseline backend (no deep learning required).

Computes an HSV color histogram as a simple global descriptor.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ColorHistogram:
    """HSV color histogram descriptor.

    A lightweight baseline that requires only OpenCV and NumPy.
    """

    def __init__(
        self,
        h_bins: int = 32,
        s_bins: int = 32,
        v_bins: int = 16,
    ) -> None:
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self._dim = h_bins + s_bins + v_bins

    @property
    def descriptor_dim(self) -> int:
        return self._dim

    def extract(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [self.v_bins], [0, 256]).flatten()

        descriptor = np.concatenate([h_hist, s_hist, v_hist])
        # L2-normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
        return descriptor

    def extract_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        return np.stack([self.extract(p) for p in image_paths])

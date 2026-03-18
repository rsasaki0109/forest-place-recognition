"""CosPlace wrapper backend.

Requires the ``cosplace`` package to be installed separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image
    from torchvision import transforms as T
    import cosplace  # noqa: F401

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def is_available() -> bool:
    """Return True if the cosplace package is importable."""
    return _AVAILABLE


class CosPlace:
    """Thin wrapper around the CosPlace model.

    Raises ``RuntimeError`` if the ``cosplace`` package is not installed.
    """

    def __init__(
        self,
        backbone: str = "ResNet50",
        fc_output_dim: int = 2048,
        device: str | None = None,
    ) -> None:
        if not _AVAILABLE:
            raise RuntimeError(
                "CosPlace is not installed. "
                "Install it with: pip install cosplace"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._dim = fc_output_dim

        from cosplace import cosplace_network

        self._model = cosplace_network.GeoLocalizationNet(
            backbone=backbone, fc_output_dim=fc_output_dim
        )
        self._model = self._model.to(self.device).eval()

        self._transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def descriptor_dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def extract(self, image_path: Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        descriptor = self._model(tensor)
        return descriptor.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def extract_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        all_descriptors: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch = image_paths[start : start + batch_size]
            tensors = torch.stack(
                [self._transform(Image.open(p).convert("RGB")) for p in batch]
            ).to(self.device)
            descs = self._model(tensors)
            all_descriptors.append(descs.cpu().numpy())
        return np.concatenate(all_descriptors, axis=0)

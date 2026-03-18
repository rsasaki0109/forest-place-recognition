"""ResNet + Generalized Mean (GeM) pooling backend.

Uses pretrained ImageNet weights -- no VPR-specific training required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GeM(nn.Module):
    """Generalized Mean Pooling.

    See *Fine-tuning CNN Image Retrieval with No Human Annotation*
    (Radenovic et al., 2018).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)


class ResNetGeM:
    """ResNet backbone with GeM pooling for global image descriptors."""

    def __init__(
        self,
        backbone: str = "resnet50",
        image_size: tuple[int, int] = (224, 224),
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        import torchvision.models as models

        weights_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        }
        if backbone not in weights_map:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(weights_map)}")

        factory, weights, self._dim = weights_map[backbone]
        resnet = factory(weights=weights)
        # Keep only convolutional layers
        self._backbone = nn.Sequential(*list(resnet.children())[:-2]).to(self.device)
        self._backbone.eval()

        self._gem = GeM().to(self.device)
        self._gem.eval()

        self._transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @property
    def descriptor_dim(self) -> int:
        return self._dim

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self._transform(img)

    @torch.no_grad()
    def extract(self, image_path: Path) -> np.ndarray:
        tensor = self._load_image(image_path).unsqueeze(0).to(self.device)
        features = self._backbone(tensor)
        descriptor = self._gem(features)
        descriptor = F.normalize(descriptor, p=2, dim=1)
        return descriptor.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def extract_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        all_descriptors: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = torch.stack([self._load_image(p) for p in batch_paths])
            tensors = tensors.to(self.device)
            features = self._backbone(tensors)
            descriptors = self._gem(features)
            descriptors = F.normalize(descriptors, p=2, dim=1)
            all_descriptors.append(descriptors.cpu().numpy())
        return np.concatenate(all_descriptors, axis=0)

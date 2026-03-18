"""Feature extraction: NetVLAD-style global descriptors.

The model itself is a stub (random projection) to be replaced with a
trained NetVLAD or similar network.  Image preprocessing uses real
torchvision transforms matching standard ImageNet-normalised pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class NetVLADStub(nn.Module):
    """Placeholder for a NetVLAD descriptor head.

    Produces a fixed-dimension descriptor by global average pooling of a
    backbone feature map followed by a linear projection.  This is *not*
    a real NetVLAD implementation -- replace with a trained model for
    production use.
    """

    def __init__(self, backbone_dim: int = 512, descriptor_dim: int = 4096) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(backbone_dim, descriptor_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project backbone features to a global descriptor.

        Parameters
        ----------
        features:
            Backbone feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            L2-normalised descriptors of shape ``(B, descriptor_dim)``.
        """
        x = self.pool(features).flatten(1)
        x = self.proj(x)
        return nn.functional.normalize(x, p=2, dim=1)


class FeatureExtractor:
    """Extract global image descriptors using a CNN backbone + NetVLAD stub."""

    def __init__(
        self,
        descriptor_dim: int = 4096,
        image_size: tuple[int, int] = (224, 224),
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.descriptor_dim = descriptor_dim

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Backbone: use ResNet-18 conv layers (pretrained weights)
        import torchvision.models as models

        resnet = models.resnet18(weights=None)  # no download in stub mode
        # Remove the final FC and avgpool -- keep conv features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]).to(self.device)
        self.backbone.eval()

        self.head = NetVLADStub(
            backbone_dim=512,
            descriptor_dim=descriptor_dim,
        ).to(self.device)
        self.head.eval()

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single image."""
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    @torch.no_grad()
    def extract(self, image_path: Path) -> np.ndarray:
        """Extract a single global descriptor from an image.

        Returns
        -------
        np.ndarray
            Descriptor vector of shape ``(descriptor_dim,)``.
        """
        tensor = self._load_image(image_path).unsqueeze(0).to(self.device)
        features = self.backbone(tensor)
        descriptor = self.head(features)
        return descriptor.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def extract_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract descriptors for a list of images.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, descriptor_dim)``.
        """
        all_descriptors: list[np.ndarray] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = torch.stack([self._load_image(p) for p in batch_paths])
            tensors = tensors.to(self.device)

            features = self.backbone(tensors)
            descriptors = self.head(features)
            all_descriptors.append(descriptors.cpu().numpy())

        return np.concatenate(all_descriptors, axis=0)

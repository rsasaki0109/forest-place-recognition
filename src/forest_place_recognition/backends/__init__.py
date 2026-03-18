"""VPR backend registry.

Each backend wraps a feature-extraction method behind a common interface
(see :class:`~.base.VPRBackend`).
"""

from __future__ import annotations

from typing import Any

# Registry: name -> (module_path, class_name, requires_optional_dep)
_BACKENDS: dict[str, tuple[str, str, bool]] = {
    "resnet_gem": (
        "forest_place_recognition.backends.resnet_gem",
        "ResNetGeM",
        False,
    ),
    "histogram": (
        "forest_place_recognition.backends.color_histogram",
        "ColorHistogram",
        False,
    ),
    "cosplace": (
        "forest_place_recognition.backends.cosplace",
        "CosPlace",
        True,
    ),
    "eigenplaces": (
        "forest_place_recognition.backends.eigenplaces",
        "EigenPlaces",
        True,
    ),
}

BACKEND_NAMES: list[str] = list(_BACKENDS.keys())


def get_backend(name: str, **kwargs: Any) -> Any:
    """Instantiate a VPR backend by name.

    Parameters
    ----------
    name:
        Backend identifier (one of ``resnet_gem``, ``histogram``,
        ``cosplace``, ``eigenplaces``).
    **kwargs:
        Keyword arguments forwarded to the backend constructor.

    Returns
    -------
    A backend instance with ``extract(image_path)`` and
    ``extract_batch(image_paths)`` methods.

    Raises
    ------
    ValueError
        If *name* is not a registered backend.
    RuntimeError
        If the backend requires an optional dependency that is not installed.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {BACKEND_NAMES}"
        )

    module_path, class_name, _ = _BACKENDS[name]
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def available_backends() -> list[str]:
    """Return names of backends whose dependencies are satisfied."""
    available: list[str] = []
    for name, (module_path, _class_name, requires_opt) in _BACKENDS.items():
        if not requires_opt:
            available.append(name)
        else:
            import importlib

            mod = importlib.import_module(module_path)
            if hasattr(mod, "is_available") and mod.is_available():
                available.append(name)
    return available

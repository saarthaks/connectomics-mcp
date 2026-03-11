"""Dataset → backend routing and capability checking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from connectomics_mcp.exceptions import DatasetNotSupported

if TYPE_CHECKING:
    from connectomics_mcp.backends.base import ConnectomeBackend

logger = logging.getLogger(__name__)

DATASETS: dict[str, dict[str, Any]] = {
    "flywire": {
        "backend": "cave",
        "class": "FlyWireBackend",
        "capabilities": ["cave", "universal"],
    },
    "minnie65": {
        "backend": "cave",
        "class": "MICrONSBackend",
        "capabilities": ["cave", "universal"],
    },
    "hemibrain": {
        "backend": "neuprint",
        "server": "neuprint.janelia.org",
        "dataset": "hemibrain:v1.2.1",
        "capabilities": ["neuprint", "universal"],
    },
}

_backend_cache: dict[str, ConnectomeBackend] = {}


def get_backend(dataset: str) -> ConnectomeBackend:
    """Return the appropriate backend instance for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. "minnie65", "flywire", "hemibrain").

    Returns
    -------
    ConnectomeBackend
        The backend instance.

    Raises
    ------
    DatasetNotSupported
        If the dataset is not recognized.
    """
    if dataset not in DATASETS:
        raise DatasetNotSupported(dataset)

    if dataset in _backend_cache:
        return _backend_cache[dataset]

    config = DATASETS[dataset]
    backend_type = config["backend"]

    if backend_type == "cave":
        import connectomics_mcp.backends.cave_backend as cave_module

        backend_cls = getattr(cave_module, config["class"])
        backend = backend_cls()
    elif backend_type == "neuprint":
        from connectomics_mcp.backends.neuprint_backend import NeuPrintBackend

        backend = NeuPrintBackend(
            server=config["server"],
            dataset=config["dataset"],
            dataset_name=dataset,
        )
    else:
        raise DatasetNotSupported(dataset)

    _backend_cache[dataset] = backend
    logger.debug("Created %s backend for dataset '%s'", backend_type, dataset)
    return backend


def check_capability(dataset: str, capability: str) -> None:
    """Verify a dataset supports a given capability.

    Parameters
    ----------
    dataset : str
        Dataset name.
    capability : str
        Required capability (e.g. "universal", "cave", "neuprint").

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or lacks the capability.
    """
    if dataset not in DATASETS:
        raise DatasetNotSupported(dataset)

    if capability not in DATASETS[dataset]["capabilities"]:
        raise DatasetNotSupported(dataset, capability)

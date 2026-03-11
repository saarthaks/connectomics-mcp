"""Neuroglancer URL builder for connectomic datasets.

Constructs fully encoded Neuroglancer URLs with EM + segmentation
layers and selected segments, using the correct coordinate space
and Neuroglancer instance for each dataset.
"""

from __future__ import annotations

import base64
import json
import logging
import zlib
from typing import Any

logger = logging.getLogger(__name__)

# Dataset-specific configuration for Neuroglancer URL building
NEUROGLANCER_CONFIGS: dict[str, dict[str, Any]] = {
    "minnie65": {
        "base_url": "https://neuroglancer.brain-map.org",
        "em_source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
        "seg_source": "graphene://https://prodv1.flywire-daf.com/segmentation/api/v1/minnie65_public",
        "voxel_size": [8, 8, 40],
        "coordinate_space": "nm",
    },
    "flywire": {
        "base_url": "https://ngl.flywire.ai",
        "em_source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
        "seg_source": "graphene://https://prodv1.flywire-daf.com/segmentation/api/v1/flywire_fafb_production",
        "voxel_size": [4, 4, 40],
        "coordinate_space": "nm",
    },
    "hemibrain": {
        "base_url": "https://neuroglancer-demo.appspot.com",
        "em_source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
        "seg_source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
        "voxel_size": [8, 8, 8],
        "coordinate_space": "nm",
    },
    "fanc": {
        "base_url": "https://neuroglancer.brain-map.org",
        "em_source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/fanc/em",
        "seg_source": "graphene://https://prodv1.flywire-daf.com/segmentation/api/v1/fanc_production_mar2021",
        "voxel_size": [4.3, 4.3, 45],
        "coordinate_space": "nm",
    },
}


def _build_state_json(
    segment_ids: list[int | str],
    dataset: str,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the Neuroglancer state JSON for the given segments.

    Parameters
    ----------
    segment_ids : list[int | str]
        Segment IDs to select in the segmentation layer.
    dataset : str
        Dataset name for config lookup.
    annotations : list[dict], optional
        Point annotations to add as an annotation layer.

    Returns
    -------
    dict
        Neuroglancer state JSON.
    """
    config = NEUROGLANCER_CONFIGS[dataset]

    layers = [
        {
            "type": "image",
            "source": config["em_source"],
            "name": "em",
        },
        {
            "type": "segmentation",
            "source": config["seg_source"],
            "name": "segmentation",
            "segments": [str(sid) for sid in segment_ids],
        },
    ]

    layer_names = ["em", "segmentation"]

    if annotations:
        layers.append(
            {
                "type": "annotation",
                "name": "annotations",
                "annotations": annotations,
            }
        )
        layer_names.append("annotations")

    state = {
        "layers": layers,
        "selectedLayer": {"layer": "segmentation", "visible": True},
        "layout": "4panel",
    }

    return state


def build_neuroglancer_url(
    segment_ids: list[int | str],
    dataset: str,
    annotations: list[dict] | None = None,
) -> str:
    """Build a Neuroglancer URL for the given segments and dataset.

    Parameters
    ----------
    segment_ids : list[int | str]
        Segment IDs to highlight in the segmentation layer.
    dataset : str
        Dataset name (must be in NEUROGLANCER_CONFIGS).
    annotations : list[dict], optional
        Point annotations to overlay.

    Returns
    -------
    str
        Complete Neuroglancer URL with encoded state.

    Raises
    ------
    KeyError
        If the dataset is not in NEUROGLANCER_CONFIGS.
    """
    if dataset not in NEUROGLANCER_CONFIGS:
        raise KeyError(f"No Neuroglancer config for dataset '{dataset}'")

    config = NEUROGLANCER_CONFIGS[dataset]
    state = _build_state_json(segment_ids, dataset, annotations)

    state_json = json.dumps(state, separators=(",", ":"))
    compressed = zlib.compress(state_json.encode("utf-8"))
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii")

    url = f"{config['base_url']}/#!{encoded}"
    logger.debug("Built Neuroglancer URL for %d segments in %s", len(segment_ids), dataset)
    return url


def get_layers_for_dataset(dataset: str) -> list[str]:
    """Return the layer names included in a Neuroglancer URL for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.

    Returns
    -------
    list[str]
        Layer names (e.g. ["em", "segmentation"]).
    """
    return ["em", "segmentation"]

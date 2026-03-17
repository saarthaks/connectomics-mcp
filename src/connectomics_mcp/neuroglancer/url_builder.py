"""Neuroglancer URL builder for connectomic datasets.

Constructs fully encoded Neuroglancer URLs with EM + segmentation
layers and selected segments, using the correct coordinate space
and Neuroglancer instance for each dataset.

Two state formats are used:
- "spelunker": nglui-compatible format with dimensions, source arrays
  with outputDimensions transforms, skeleton source, middleauth.
  Used by spelunker.cave-explorer.org (minnie65).
- "flywire": Simpler format with plain string sources and
  segmentation_with_graph layer type. Used by ngl.flywire.ai.
- "compressed": zlib+base64 encoded state for standard neuroglancer
  instances. Used by neuroglancer-demo.appspot.com (hemibrain).
"""

from __future__ import annotations

import base64
import json
import logging
import zlib
from typing import Any

logger = logging.getLogger(__name__)


def _source_with_transform(
    url: str,
    dimensions: dict[str, list],
) -> dict[str, Any]:
    """Wrap a source URL with an outputDimensions transform."""
    return {
        "url": url,
        "transform": {"outputDimensions": dimensions},
        "subsources": {},
        "enableDefaultSubsources": True,
    }


# Dataset-specific configuration for Neuroglancer URL building.
# Dimensions use SI meters: [scale, "m"].
NEUROGLANCER_CONFIGS: dict[str, dict[str, Any]] = {
    "minnie65": {
        "base_url": "https://spelunker.cave-explorer.org",
        "em_source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
        "seg_source": "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie65_public",
        "skeleton_source": "precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_public/precomputed/skeleton/",
        "dimensions": {
            "x": [4e-9, "m"],
            "y": [4e-9, "m"],
            "z": [40e-9, "m"],
        },
        "voxel_size": [4, 4, 40],
        "coordinate_space": "nm",
        "state_format": "spelunker",
        "projection_scale": 50000.0,
    },
    "flywire": {
        "base_url": "https://ngl.flywire.ai",
        "em_source": "precomputed://gs://flywire_em/aligned/v1",
        "seg_source": "graphene://https://prod.flywire-daf.com/segmentation/1.0/flywire_public",
        "voxel_size": [4, 4, 40],
        "coordinate_space": "nm",
        "state_format": "flywire",
    },
    "hemibrain": {
        "base_url": "https://neuroglancer-demo.appspot.com",
        "em_source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
        "seg_source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
        "dimensions": {
            "x": [8e-9, "m"],
            "y": [8e-9, "m"],
            "z": [8e-9, "m"],
        },
        "voxel_size": [8, 8, 8],
        "coordinate_space": "nm",
        "state_format": "compressed",
        "projection_scale": 30000.0,
    },
}


def _build_spelunker_state(
    segment_ids: list[int | str],
    config: dict[str, Any],
    annotations: list[dict] | None = None,
    position: list[float] | None = None,
) -> dict[str, Any]:
    """Build nglui-compatible state for spelunker.cave-explorer.org."""
    dims = config["dimensions"]

    image_layer: dict[str, Any] = {
        "type": "image",
        "source": [_source_with_transform(config["em_source"], dims)],
        "name": "imagery",
    }

    seg_sources = [_source_with_transform(config["seg_source"], dims)]
    if config.get("skeleton_source"):
        seg_sources.append(
            _source_with_transform(config["skeleton_source"], dims)
        )

    seg_layer: dict[str, Any] = {
        "type": "segmentation",
        "source": seg_sources,
        "name": "segmentation",
        "segments": [str(sid) for sid in segment_ids],
        "selectedAlpha": 0.5,
        "notSelectedAlpha": 0.0,
        "objectAlpha": 1.0,
    }

    layers: list[dict[str, Any]] = [image_layer, seg_layer]

    if annotations:
        layers.append({
            "type": "annotation",
            "name": "annotations",
            "annotations": annotations,
        })

    state: dict[str, Any] = {
        "dimensions": dims,
        "crossSectionScale": 1.0,
        "projectionScale": config.get("projection_scale", 50000.0),
        "showSlices": False,
        "layers": layers,
        "layout": "3d",
    }

    if position is not None:
        state["position"] = position

    return state


def _build_flywire_state(
    segment_ids: list[int | str],
    config: dict[str, Any],
    annotations: list[dict] | None = None,
    position: list[float] | None = None,
) -> dict[str, Any]:
    """Build simple state for ngl.flywire.ai.

    FlyWire's neuroglancer fork expects:
    - segmentation_with_graph layer type
    - Plain string sources (no transform arrays)
    - No dimensions block
    """
    seg_layer: dict[str, Any] = {
        "type": "segmentation_with_graph",
        "source": config["seg_source"],
        "name": "segmentation",
        "segments": [str(sid) for sid in segment_ids],
    }

    layers: list[dict[str, Any]] = [seg_layer]

    if annotations:
        layers.append({
            "type": "annotation",
            "name": "annotations",
            "annotations": annotations,
        })

    state: dict[str, Any] = {
        "layers": layers,
        "selectedLayer": {"layer": "segmentation", "visible": True},
        "layout": "3d",
    }

    if position is not None:
        state["navigation"] = {
            "pose": {
                "position": {
                    "voxelSize": config["voxel_size"],
                    "voxelCoordinates": [
                        position[0] / config["voxel_size"][0],
                        position[1] / config["voxel_size"][1],
                        position[2] / config["voxel_size"][2],
                    ],
                }
            },
        }

    return state


def _build_compressed_state(
    segment_ids: list[int | str],
    config: dict[str, Any],
    annotations: list[dict] | None = None,
    position: list[float] | None = None,
) -> dict[str, Any]:
    """Build state for standard neuroglancer (hemibrain)."""
    dims = config.get("dimensions", {})

    layers: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": [_source_with_transform(config["em_source"], dims)],
            "name": "imagery",
        },
        {
            "type": "segmentation",
            "source": [_source_with_transform(config["seg_source"], dims)],
            "name": "segmentation",
            "segments": [str(sid) for sid in segment_ids],
        },
    ]

    if annotations:
        layers.append({
            "type": "annotation",
            "name": "annotations",
            "annotations": annotations,
        })

    state: dict[str, Any] = {
        "dimensions": dims,
        "projectionScale": config.get("projection_scale", 30000.0),
        "showSlices": False,
        "layers": layers,
        "layout": "3d",
    }

    if position is not None:
        state["position"] = position

    return state


# Dispatch table for state builders
_STATE_BUILDERS = {
    "spelunker": _build_spelunker_state,
    "flywire": _build_flywire_state,
    "compressed": _build_compressed_state,
}


def _build_state_json(
    segment_ids: list[int | str],
    dataset: str,
    annotations: list[dict] | None = None,
    position: list[float] | None = None,
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
    position : list[float], optional
        3D position to center the view on. Format depends on dataset:
        voxel coordinates for spelunker/compressed, voxel coordinates
        for flywire (converted internally).

    Returns
    -------
    dict
        Neuroglancer state JSON.
    """
    config = NEUROGLANCER_CONFIGS[dataset]
    fmt = config["state_format"]
    builder = _STATE_BUILDERS[fmt]
    return builder(segment_ids, config, annotations, position)


def build_neuroglancer_url(
    segment_ids: list[int | str],
    dataset: str,
    annotations: list[dict] | None = None,
    position: list[float] | None = None,
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
    position : list[float], optional
        3D position to center the view on.

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
    state = _build_state_json(segment_ids, dataset, annotations, position)
    state_json = json.dumps(state, separators=(",", ":"))

    fmt = config["state_format"]
    if fmt == "compressed":
        compressed = zlib.compress(state_json.encode("utf-8"))
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        url = f"{config['base_url']}/#!{encoded}"
    else:
        url = f"{config['base_url']}/#!{state_json}"

    logger.debug(
        "Built Neuroglancer URL for %d segments in %s",
        len(segment_ids),
        dataset,
    )
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
        Layer names (e.g. ["imagery", "segmentation"]).
    """
    fmt = NEUROGLANCER_CONFIGS.get(dataset, {}).get("state_format", "")
    if fmt == "flywire":
        return ["segmentation"]
    return ["imagery", "segmentation"]

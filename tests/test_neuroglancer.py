"""Tests for the Neuroglancer URL builder."""

from __future__ import annotations

import base64
import json
import zlib

import pytest

from connectomics_mcp.neuroglancer.url_builder import (
    build_neuroglancer_url,
    get_layers_for_dataset,
    NEUROGLANCER_CONFIGS,
)


class TestBuildNeuroglancerUrl:
    def test_minnie65_url(self):
        url = build_neuroglancer_url([720575940621039145], "minnie65")
        assert url.startswith("https://spelunker.cave-explorer.org/#!")

    def test_hemibrain_url(self):
        url = build_neuroglancer_url([12345], "hemibrain")
        assert url.startswith("https://neuroglancer-demo.appspot.com/#!")

    def test_flywire_url(self):
        url = build_neuroglancer_url([720575940621039145], "flywire")
        assert url.startswith("https://ngl.flywire.ai/#!")

    # --- Spelunker (minnie65) format ---

    def test_minnie65_json_decodable(self):
        url = build_neuroglancer_url([123, 456], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "layers" in state
        assert "dimensions" in state
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]
        # Should have graphene + skeleton sources as array of objects
        assert len(seg_layer["source"]) == 2
        assert "skeleton" in seg_layer["source"][1]["url"]

    def test_minnie65_has_transforms(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        src = seg_layer["source"][0]
        assert "transform" in src
        assert "outputDimensions" in src["transform"]

    def test_minnie65_dimensions(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert state["dimensions"]["x"] == [4e-9, "m"]

    def test_minnie65_position(self):
        pos = [240640.0, 207872.0, 21360.0]
        url = build_neuroglancer_url([123], "minnie65", position=pos)
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert state["position"] == pos

    # --- FlyWire format ---

    def test_flywire_json_decodable(self):
        url = build_neuroglancer_url([123, 456], "flywire")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "layers" in state
        seg_layer = [l for l in state["layers"]
                     if "segmentation" in l["type"]][0]
        assert seg_layer["type"] == "segmentation_with_graph"
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]

    def test_flywire_simple_source(self):
        """FlyWire sources are plain strings, not arrays."""
        url = build_neuroglancer_url([123], "flywire")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        seg_layer = state["layers"][0]
        assert isinstance(seg_layer["source"], str)

    def test_flywire_no_dimensions(self):
        """FlyWire state has no dimensions block."""
        url = build_neuroglancer_url([123], "flywire")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "dimensions" not in state

    def test_flywire_no_em_layer(self):
        """FlyWire only has segmentation layer (no imagery)."""
        url = build_neuroglancer_url([123], "flywire")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        layer_types = [l["type"] for l in state["layers"]]
        assert "image" not in layer_types

    def test_flywire_position(self):
        pos = [100000.0, 200000.0, 30000.0]
        url = build_neuroglancer_url([123], "flywire", position=pos)
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "navigation" in state
        voxel_coords = state["navigation"]["pose"]["position"]["voxelCoordinates"]
        # flywire voxel_size = [4, 4, 40]
        assert voxel_coords[0] == pytest.approx(25000.0)
        assert voxel_coords[1] == pytest.approx(50000.0)
        assert voxel_coords[2] == pytest.approx(750.0)

    # --- Hemibrain format ---

    def test_hemibrain_compressed_decodable(self):
        url = build_neuroglancer_url([123, 456], "hemibrain")
        fragment = url.split("#!")[1]
        decoded = zlib.decompress(base64.urlsafe_b64decode(fragment))
        state = json.loads(decoded)
        assert "layers" in state
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]

    # --- Common ---

    def test_3d_layout_default(self):
        for ds in ["minnie65", "flywire", "hemibrain"]:
            url = build_neuroglancer_url([123], ds)
            fragment = url.split("#!")[1]
            if ds == "hemibrain":
                decoded = zlib.decompress(base64.urlsafe_b64decode(fragment))
                state = json.loads(decoded)
            else:
                state = json.loads(fragment)
            assert state["layout"] == "3d", f"Failed for {ds}"

    def test_no_position_by_default(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "position" not in state

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            build_neuroglancer_url([123], "nonexistent")

    def test_get_layers_minnie65(self):
        layers = get_layers_for_dataset("minnie65")
        assert "imagery" in layers
        assert "segmentation" in layers

    def test_get_layers_flywire(self):
        layers = get_layers_for_dataset("flywire")
        assert layers == ["segmentation"]

    def test_all_datasets_have_configs(self):
        for dataset in ["minnie65", "flywire", "hemibrain"]:
            assert dataset in NEUROGLANCER_CONFIGS

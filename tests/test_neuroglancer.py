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

    def test_minnie65_url_json_decodable(self):
        url = build_neuroglancer_url([123, 456], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "layers" in state
        assert "dimensions" in state
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]
        # Should have graphene + skeleton sources
        assert len(seg_layer["source"]) == 2
        assert "skeleton" in seg_layer["source"][1]["url"]

    def test_hemibrain_url_compressed_decodable(self):
        url = build_neuroglancer_url([123, 456], "hemibrain")
        fragment = url.split("#!")[1]
        decoded = zlib.decompress(base64.urlsafe_b64decode(fragment))
        state = json.loads(decoded)
        assert "layers" in state
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]

    def test_flywire_url_json_decodable(self):
        url = build_neuroglancer_url([123, 456], "flywire")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "layers" in state
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]
        # FlyWire has no skeleton source
        assert len(seg_layer["source"]) == 1

    def test_3d_layout_default(self):
        url = build_neuroglancer_url([123], "flywire")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert state["layout"] == "3d"

    def test_dimensions_present(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "dimensions" in state
        assert "x" in state["dimensions"]
        assert state["dimensions"]["x"] == [4e-9, "m"]

    def test_position_parameter(self):
        pos = [240640.0, 207872.0, 21360.0]
        url = build_neuroglancer_url([123], "minnie65", position=pos)
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert state["position"] == pos

    def test_no_position_by_default(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert "position" not in state

    def test_projection_scale_present(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert state["projectionScale"] == 50000.0

    def test_show_slices_false(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        assert state["showSlices"] is False

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            build_neuroglancer_url([123], "nonexistent")

    def test_get_layers(self):
        layers = get_layers_for_dataset("minnie65")
        assert "imagery" in layers
        assert "segmentation" in layers

    def test_all_datasets_have_configs(self):
        for dataset in ["minnie65", "flywire", "hemibrain"]:
            assert dataset in NEUROGLANCER_CONFIGS

    def test_source_has_transform(self):
        url = build_neuroglancer_url([123], "minnie65")
        fragment = url.split("#!")[1]
        state = json.loads(fragment)
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        src = seg_layer["source"][0]
        assert "transform" in src
        assert "outputDimensions" in src["transform"]

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
        assert url.startswith("https://neuroglancer.brain-map.org/#!")

    def test_hemibrain_url(self):
        url = build_neuroglancer_url([12345], "hemibrain")
        assert url.startswith("https://neuroglancer-demo.appspot.com/#!")

    def test_flywire_url(self):
        url = build_neuroglancer_url([720575940621039145], "flywire")
        assert url.startswith("https://ngl.flywire.ai/#!")

    def test_url_decodable(self):
        url = build_neuroglancer_url([123, 456], "minnie65")
        fragment = url.split("#!")[1]
        decoded = zlib.decompress(base64.urlsafe_b64decode(fragment))
        state = json.loads(decoded)
        assert "layers" in state
        seg_layer = [l for l in state["layers"] if l["type"] == "segmentation"][0]
        assert "123" in seg_layer["segments"]
        assert "456" in seg_layer["segments"]

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            build_neuroglancer_url([123], "nonexistent")

    def test_get_layers(self):
        layers = get_layers_for_dataset("minnie65")
        assert "em" in layers
        assert "segmentation" in layers

    def test_all_datasets_have_configs(self):
        for dataset in ["minnie65", "flywire", "hemibrain", "fanc"]:
            assert dataset in NEUROGLANCER_CONFIGS

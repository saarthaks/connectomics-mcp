"""Tests for search_cell_types tool and improved FlyWire get_neurons_by_type matching."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from connectomics_mcp.output_contracts.schemas import (
    CellTypeSearchResponse,
    CellTypeTaxonomyResponse,
    NeuronsByTypeResponse,
)
from connectomics_mcp.tools.universal import (
    get_cell_type_taxonomy,
    get_neurons_by_type,
    search_cell_types,
)


# ---------------------------------------------------------------------------
# search_cell_types tests
# ---------------------------------------------------------------------------


class TestSearchCellTypes:
    """Tests for the search_cell_types discovery tool."""

    def test_flywire_exact_match(self, mock_flywire_backend):
        result = search_cell_types("EPG", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.dataset == "flywire"
        assert resp.query == "EPG"
        assert resp.n_matches > 0
        # Should find EPG at both cell_type and tag levels
        cell_type_matches = [
            m for m in resp.matches if m.cell_type == "EPG"
        ]
        assert len(cell_type_matches) > 0

    def test_flywire_tag_level_match(self, mock_flywire_backend):
        """Searching 'EPG' should find the tag-level match (most specific)."""
        result = search_cell_types("EPG", "flywire")
        resp = CellTypeSearchResponse(**result)
        tag_matches = [
            m for m in resp.matches
            if m.cell_type == "EPG" and m.classification_level == "tag"
        ]
        assert len(tag_matches) > 0
        # Tag level should be sorted first (most specific)
        assert resp.matches[0].classification_level == "tag"

    def test_flywire_substring_match(self, mock_flywire_backend):
        result = search_cell_types("PN", "flywire")
        resp = CellTypeSearchResponse(**result)
        # Should find DA1_lPN (contains "PN") and PEN_a, PEN_b
        assert resp.n_matches > 0
        matched_types = [m.cell_type for m in resp.matches]
        assert any("PN" in ct or "PN" in ct.upper() for ct in matched_types)

    def test_flywire_case_insensitive(self, mock_flywire_backend):
        result = search_cell_types("epg", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches > 0
        cell_type_matches = [m for m in resp.matches if m.cell_type == "EPG"]
        assert len(cell_type_matches) > 0

    def test_flywire_cross_level_search(self, mock_flywire_backend):
        """Search should find matches across classification levels."""
        result = search_cell_types("compass", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches > 0
        # "compass" should match at cell_sub_class level
        levels = [m.classification_level for m in resp.matches]
        assert "cell_sub_class" in levels

    def test_flywire_no_match(self, mock_flywire_backend):
        result = search_cell_types("NONEXISTENT_TYPE_XYZ", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches == 0
        assert resp.matches == []

    def test_hemibrain_search(self, mock_neuprint_backend):
        result = search_cell_types("EPG", "hemibrain")
        resp = CellTypeSearchResponse(**result)
        assert resp.dataset == "hemibrain"
        assert resp.n_matches > 0
        matched_types = [m.cell_type for m in resp.matches]
        assert "EPG" in matched_types

    def test_hemibrain_substring(self, mock_neuprint_backend):
        result = search_cell_types("KC", "hemibrain")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches > 0
        matched_types = [m.cell_type for m in resp.matches]
        assert any("KC" in ct for ct in matched_types)

    def test_minnie65_search(self, mock_cave_backend):
        result = search_cell_types("L2", "minnie65")
        resp = CellTypeSearchResponse(**result)
        assert resp.dataset == "minnie65"
        assert resp.n_matches > 0
        matched_types = [m.cell_type for m in resp.matches]
        assert any("L2" in ct for ct in matched_types)

    def test_matches_include_neuron_counts(self, mock_flywire_backend):
        result = search_cell_types("EPG", "flywire")
        resp = CellTypeSearchResponse(**result)
        for match in resp.matches:
            assert match.n_neurons > 0

    def test_matches_include_classification_level(self, mock_flywire_backend):
        result = search_cell_types("central", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches > 0
        for match in resp.matches:
            assert match.classification_level is not None


# ---------------------------------------------------------------------------
# FlyWire get_neurons_by_type progressive matching tests
# ---------------------------------------------------------------------------


class TestFlyWireProgressiveMatching:
    """Tests for improved FlyWire get_neurons_by_type with progressive fallback."""

    def test_exact_cell_type_level_match(self, mock_flywire_backend, tmp_path):
        """Exact match at cell_type level should work as before."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("EPG", "flywire")
        resp = NeuronsByTypeResponse(**result)
        assert resp.n_total == 2  # Two EPG neurons in mock data
        assert resp.dataset == "flywire"

    def test_exact_match_any_level(self, mock_flywire_backend, tmp_path):
        """Exact match at non-cell_type level (e.g., cell_sub_class='compass')."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("compass", "flywire")
        resp = NeuronsByTypeResponse(**result)
        # "compass" is a cell_sub_class value, not cell_type
        assert resp.n_total > 0
        assert any("not found at cell_type level" in w for w in resp.warnings)

    def test_case_insensitive_match(self, mock_flywire_backend, tmp_path):
        """Case-insensitive matching should find EPG from 'epg'."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("epg", "flywire")
        resp = NeuronsByTypeResponse(**result)
        assert resp.n_total > 0
        assert any("case-insensitively" in w for w in resp.warnings)

    def test_substring_match(self, mock_flywire_backend, tmp_path):
        """Substring matching should find PEN_a and PEN_b from 'PEN'."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("PEN", "flywire")
        resp = NeuronsByTypeResponse(**result)
        # PEN matches PEN_a and PEN_b via substring
        assert resp.n_total > 0
        assert any("substring" in w for w in resp.warnings)

    def test_no_match_suggests_search(self, mock_flywire_backend, tmp_path):
        """When nothing matches, warning should suggest search_cell_types."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("NONEXISTENT_XYZ", "flywire")
        resp = NeuronsByTypeResponse(**result)
        assert resp.n_total == 0
        assert any("search_cell_types" in w for w in resp.warnings)

    def test_exact_match_no_warnings(self, mock_flywire_backend, tmp_path):
        """Exact match at cell_type level should have no matching warnings."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("Delta7", "flywire")
        resp = NeuronsByTypeResponse(**result)
        assert resp.n_total == 2  # Two Delta7 neurons in mock data
        # No fuzzy matching warnings
        assert not any("substring" in w for w in resp.warnings)
        assert not any("case-insensitively" in w for w in resp.warnings)

    def test_region_filter_still_works(self, mock_flywire_backend, tmp_path):
        """Region filter should apply after matching."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("EPG", "flywire", region="central")
        resp = NeuronsByTypeResponse(**result)
        assert resp.n_total > 0

    def test_artifact_written(self, mock_flywire_backend, tmp_path):
        """Artifact should be written even when using fuzzy match."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            result = get_neurons_by_type("EPG", "flywire")
        resp = NeuronsByTypeResponse(**result)
        assert resp.artifact_manifest is not None
        artifact_path = Path(resp.artifact_manifest.artifact_path)
        assert artifact_path.exists()
        df = pd.read_parquet(artifact_path)
        assert len(df) == resp.n_total


# ---------------------------------------------------------------------------
# get_cell_type_taxonomy tests
# ---------------------------------------------------------------------------


class TestGetCellTypeTaxonomy:
    """Tests for the get_cell_type_taxonomy tool."""

    def test_flywire_has_5_levels(self, mock_flywire_backend):
        """FlyWire has 4 hierarchy levels + tag level from neuron_information_v2."""
        result = get_cell_type_taxonomy("flywire")
        resp = CellTypeTaxonomyResponse(**result)
        assert resp.dataset == "flywire"
        assert len(resp.levels) == 5
        level_names = [lv.level_name for lv in resp.levels]
        assert "super_class" in level_names
        assert "cell_class" in level_names
        assert "cell_sub_class" in level_names
        assert "cell_type" in level_names
        assert "tag" in level_names

    def test_flywire_levels_have_values(self, mock_flywire_backend):
        result = get_cell_type_taxonomy("flywire")
        resp = CellTypeTaxonomyResponse(**result)
        for level in resp.levels:
            assert len(level.values) > 0
            for v in level.values:
                assert "name" in v
                assert "n_neurons" in v
                assert v["n_neurons"] > 0

    def test_flywire_cell_class_contains_cx(self, mock_flywire_backend):
        """CX should appear as a cell_class value."""
        result = get_cell_type_taxonomy("flywire")
        resp = CellTypeTaxonomyResponse(**result)
        class_level = next(
            lv for lv in resp.levels if lv.level_name == "cell_class"
        )
        class_names = [v["name"] for v in class_level.values]
        # Mock data has "olfactory" and "central_complex"
        assert any("olfactory" in n or "central_complex" in n for n in class_names)

    def test_flywire_tag_level_has_specific_types(self, mock_flywire_backend):
        """Tag level should have specific cell type names like EPG, PEN_a."""
        result = get_cell_type_taxonomy("flywire")
        resp = CellTypeTaxonomyResponse(**result)
        tag_level = next(lv for lv in resp.levels if lv.level_name == "tag")
        tag_names = [v["name"] for v in tag_level.values]
        assert "EPG" in tag_names
        assert "PEN_a" in tag_names
        assert "Delta7" in tag_names

    def test_flywire_example_lineages(self, mock_flywire_backend):
        result = get_cell_type_taxonomy("flywire")
        resp = CellTypeTaxonomyResponse(**result)
        assert len(resp.example_lineages) > 0
        for lineage in resp.example_lineages:
            # Should have at least cell_class and cell_type
            assert len(lineage) >= 2

    def test_flywire_n_total_neurons(self, mock_flywire_backend):
        result = get_cell_type_taxonomy("flywire")
        resp = CellTypeTaxonomyResponse(**result)
        assert resp.n_total_neurons > 0

    def test_hemibrain_flat_taxonomy(self, mock_neuprint_backend):
        result = get_cell_type_taxonomy("hemibrain")
        resp = CellTypeTaxonomyResponse(**result)
        assert resp.dataset == "hemibrain"
        assert len(resp.levels) == 1
        assert resp.levels[0].level_name == "type"
        type_names = [v["name"] for v in resp.levels[0].values]
        assert "EPG" in type_names
        assert "Delta7" in type_names

    def test_minnie65_flat_taxonomy(self, mock_cave_backend):
        result = get_cell_type_taxonomy("minnie65")
        resp = CellTypeTaxonomyResponse(**result)
        assert resp.dataset == "minnie65"
        assert len(resp.levels) == 1
        assert resp.levels[0].level_name == "cell_type"


# ---------------------------------------------------------------------------
# search_cell_types taxonomy_hints tests
# ---------------------------------------------------------------------------


class TestSearchTaxonomyHints:
    """Tests that 0-result searches include taxonomy hints."""

    def test_flywire_0_results_has_hints(self, mock_flywire_backend):
        result = search_cell_types("NONEXISTENT_XYZ", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches == 0
        assert len(resp.taxonomy_hints) > 0
        hint = resp.taxonomy_hints[0]
        assert "cell_class" in hint
        assert "get_cell_type_taxonomy" in hint

    def test_flywire_match_has_no_hints(self, mock_flywire_backend):
        result = search_cell_types("EPG", "flywire")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches > 0
        assert len(resp.taxonomy_hints) == 0

    def test_minnie65_0_results_has_hints(self, mock_cave_backend):
        result = search_cell_types("NONEXISTENT_XYZ", "minnie65")
        resp = CellTypeSearchResponse(**result)
        assert resp.n_matches == 0
        assert len(resp.taxonomy_hints) > 0
        assert "get_cell_type_taxonomy" in resp.taxonomy_hints[0]

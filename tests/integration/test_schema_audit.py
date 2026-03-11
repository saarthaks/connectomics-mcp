"""Schema audit tests — introspect real API table schemas.

Run with:  pytest tests/integration/test_schema_audit.py --integration -v -s

These tests query each table/API with minimal data (limit=1 or small queries)
and print + assert on the actual columns returned. Results feed backend fixes.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


# ── minnie65 (CAVE) ─────────────────────────────────────────────────


class TestMinnie65Schema:
    """Introspect minnie65_public table schemas."""

    def test_cell_type_table(self, cave_client_minnie65) -> None:
        """Check aibs_metamodel_celltypes_v661 columns."""
        c = cave_client_minnie65
        df = c.materialize.query_table(
            "aibs_metamodel_celltypes_v661",
            limit=1,
        )
        print(f"\n  aibs_metamodel_celltypes_v661 columns: {list(df.columns)}")
        print(f"  dtypes:\n{df.dtypes}")
        if not df.empty:
            print(f"  sample row:\n{df.iloc[0].to_dict()}")

        # Assert critical columns our code depends on
        assert "pt_root_id" in df.columns
        assert "cell_type" in df.columns

    def test_synapse_table(self, cave_client_minnie65) -> None:
        """Check synapses_pni_2 columns."""
        c = cave_client_minnie65
        df = c.materialize.query_table(
            "synapses_pni_2",
            limit=1,
        )
        print(f"\n  synapses_pni_2 columns: {list(df.columns)}")
        print(f"  dtypes:\n{df.dtypes}")

        assert "pre_pt_root_id" in df.columns
        assert "post_pt_root_id" in df.columns

    def test_nucleus_table(self, cave_client_minnie65) -> None:
        """Check nucleus_detection_v0 columns."""
        c = cave_client_minnie65
        df = c.materialize.query_table(
            "nucleus_detection_v0",
            limit=1,
        )
        print(f"\n  nucleus_detection_v0 columns: {list(df.columns)}")
        print(f"  dtypes:\n{df.dtypes}")
        if not df.empty:
            print(f"  sample row:\n{df.iloc[0].to_dict()}")

        assert "id" in df.columns
        assert "pt_root_id" in df.columns

    def test_proofreading_table(self, cave_client_minnie65) -> None:
        """Check proofreading_status_and_strategy columns."""
        c = cave_client_minnie65
        df = c.materialize.query_table(
            "proofreading_status_and_strategy",
            limit=1,
        )
        print(f"\n  proofreading_status_and_strategy columns: {list(df.columns)}")
        print(f"  dtypes:\n{df.dtypes}")
        if not df.empty:
            print(f"  sample row:\n{df.iloc[0].to_dict()}")

        assert "pt_root_id" in df.columns
        assert "status_axon" in df.columns
        assert "status_dendrite" in df.columns

    def test_changelog_return_type(self, cave_client_minnie65) -> None:
        """Verify get_tabular_change_log returns dict[int, DataFrame]."""
        c = cave_client_minnie65
        # Use a known current root ID
        test_root = 864691135571546917
        result = c.chunkedgraph.get_tabular_change_log([test_root])

        print(f"\n  get_tabular_change_log return type: {type(result)}")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

        if test_root in result:
            changelog = result[test_root]
            print(f"  changelog type: {type(changelog)}")
            assert isinstance(changelog, pd.DataFrame)
            print(f"  changelog columns: {list(changelog.columns)}")
            print(f"  changelog dtypes:\n{changelog.dtypes}")
            if not changelog.empty:
                print(f"  sample row:\n{changelog.iloc[0].to_dict()}")
                # Assert expected columns
                assert "is_merge" in changelog.columns
                assert "user_id" in changelog.columns
        else:
            print(f"  root_id {test_root} not in result keys: {list(result.keys())}")


# ── flywire (CAVE) ──────────────────────────────────────────────────


class TestFlyWireSchema:
    """Introspect flywire_fafb_public table schemas."""

    def test_list_tables(self, cave_client_flywire) -> None:
        """Document available tables."""
        c = cave_client_flywire
        try:
            tables = c.materialize.get_tables()
            print(f"\n  flywire available tables ({len(tables)}):")
            for t in sorted(tables)[:30]:  # Print first 30
                print(f"    {t}")
            if len(tables) > 30:
                print(f"    ... and {len(tables) - 30} more")
        except Exception as e:
            if "missing_permission" in str(e) or "FORBIDDEN" in str(e):
                pytest.skip(f"FlyWire access denied: {e}")
            raise

    def test_cell_type_table(self, cave_client_flywire) -> None:
        """Check neuron_information_v2 table columns (FlyWire's cell type table)."""
        c = cave_client_flywire
        try:
            df = c.materialize.query_table(
                "neuron_information_v2",
                limit=1,
            )
            print(f"\n  neuron_information_v2 columns: {list(df.columns)}")
            print(f"  dtypes:\n{df.dtypes}")
            if not df.empty:
                print(f"  sample row:\n{df.iloc[0].to_dict()}")
            assert "pt_root_id" in df.columns
            assert "tag" in df.columns  # FlyWire uses 'tag' instead of 'cell_type'
        except Exception as e:
            if "missing_permission" in str(e) or "FORBIDDEN" in str(e):
                pytest.skip(f"FlyWire access denied: {e}")
            raise

    def test_synapse_table(self, cave_client_flywire) -> None:
        """Check synapses_nt_v1 table columns."""
        c = cave_client_flywire
        try:
            df = c.materialize.query_table(
                "synapses_nt_v1",
                limit=1,
            )
            print(f"\n  synapses_nt_v1 columns: {list(df.columns)}")
            print(f"  dtypes:\n{df.dtypes}")
            assert "pre_pt_root_id" in df.columns
            assert "post_pt_root_id" in df.columns
        except Exception as e:
            if "missing_permission" in str(e) or "FORBIDDEN" in str(e):
                pytest.skip(f"FlyWire access denied: {e}")
            raise

    def test_nucleus_table(self, cave_client_flywire) -> None:
        """Check nuclei_v1 table columns."""
        c = cave_client_flywire
        try:
            df = c.materialize.query_table(
                "nuclei_v1",
                limit=1,
            )
            print(f"\n  nuclei_v1 columns: {list(df.columns)}")
            print(f"  dtypes:\n{df.dtypes}")
            if not df.empty:
                print(f"  sample row:\n{df.iloc[0].to_dict()}")
        except Exception as e:
            if "missing_permission" in str(e) or "FORBIDDEN" in str(e):
                pytest.skip(f"FlyWire access denied: {e}")
            raise

    def test_proofreading_table(self, cave_client_flywire) -> None:
        """Check proofread_neurons table columns."""
        c = cave_client_flywire
        try:
            df = c.materialize.query_table(
                "proofread_neurons",
                limit=1,
            )
            print(f"\n  proofread_neurons columns: {list(df.columns)}")
            print(f"  dtypes:\n{df.dtypes}")
            if not df.empty:
                print(f"  sample row:\n{df.iloc[0].to_dict()}")
            assert "pt_root_id" in df.columns
        except Exception as e:
            if "missing_permission" in str(e) or "FORBIDDEN" in str(e):
                pytest.skip(f"FlyWire access denied: {e}")
            raise



# ── hemibrain (neuPrint) ─────────────────────────────────────────────


class TestHemibrainSchema:
    """Introspect hemibrain neuPrint schemas."""

    BODY_ID = 5813105172  # DA1 adPN

    def test_fetch_neurons_columns(self, neuprint_client) -> None:
        """Check fetch_neurons return schemas and somaLocation type."""
        from neuprint import NeuronCriteria as NC, fetch_neurons

        neuron_df, roi_df = fetch_neurons(NC(bodyId=self.BODY_ID))

        print(f"\n  neuron_df columns: {list(neuron_df.columns)}")
        print(f"  neuron_df dtypes:\n{neuron_df.dtypes}")
        if not neuron_df.empty:
            row = neuron_df.iloc[0]
            soma_loc = row.get("somaLocation", None)
            print(f"  somaLocation value: {soma_loc}")
            print(f"  somaLocation type: {type(soma_loc)}")
            if soma_loc is not None:
                assert isinstance(soma_loc, (list, type(None))), (
                    f"somaLocation is {type(soma_loc)}, expected list"
                )

        print(f"\n  roi_df columns: {list(roi_df.columns)}")
        print(f"  roi_df dtypes:\n{roi_df.dtypes}")
        if not roi_df.empty:
            print(f"  roi_df sample:\n{roi_df.head(3).to_dict()}")

        # Assert critical columns
        assert "bodyId" in neuron_df.columns
        assert "type" in neuron_df.columns
        assert "pre" in neuron_df.columns
        assert "post" in neuron_df.columns

    def test_fetch_adjacencies_columns(self, neuprint_client) -> None:
        """Check fetch_adjacencies return tuple and conn_df columns."""
        from neuprint import NeuronCriteria as NC, fetch_adjacencies

        # Small query: just one target
        neuron_df, conn_df = fetch_adjacencies(
            NC(), NC(bodyId=self.BODY_ID)
        )

        print(f"\n  fetch_adjacencies returns: ({type(neuron_df).__name__}, {type(conn_df).__name__})")
        print(f"  neuron_df columns: {list(neuron_df.columns)}")
        print(f"  conn_df columns: {list(conn_df.columns)}")
        print(f"  conn_df dtypes:\n{conn_df.dtypes}")
        if not conn_df.empty:
            print(f"  conn_df sample:\n{conn_df.head(3).to_dict()}")

        # Assert critical columns
        assert "bodyId_pre" in conn_df.columns
        assert "bodyId_post" in conn_df.columns
        assert "weight" in conn_df.columns

    def test_roi_connectivity_via_cypher(self, neuprint_client) -> None:
        """Verify ROI connectivity Cypher query returns expected columns.

        neuprint-python has no ``fetch_roi_connectivity`` function;
        our backend uses a Cypher query instead.
        """
        cypher = """\
            MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
            WITH n, m, e, apoc.convert.fromJsonMap(e.roiInfo) AS roiInfo
            UNWIND keys(roiInfo) AS roi
            WITH roi, roiInfo[roi] AS roiData
            WHERE roiData.post IS NOT NULL AND roiData.post > 0
            RETURN roi AS from_roi, roi AS to_roi,
                   sum(roiData.post) AS n_synapses,
                   count(*) AS n_connections
            LIMIT 5
        """
        roi_conn_df = neuprint_client.fetch_custom(cypher)

        print(f"\n  ROI connectivity Cypher result columns: {list(roi_conn_df.columns)}")
        print(f"  dtypes:\n{roi_conn_df.dtypes}")
        if not roi_conn_df.empty:
            print(f"  sample:\n{roi_conn_df.head(3).to_dict()}")

        assert "from_roi" in roi_conn_df.columns
        assert "to_roi" in roi_conn_df.columns
        assert "n_synapses" in roi_conn_df.columns
        assert "n_connections" in roi_conn_df.columns

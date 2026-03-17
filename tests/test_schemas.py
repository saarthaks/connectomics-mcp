"""Verify all Pydantic schemas instantiate and serialize correctly."""

from connectomics_mcp.output_contracts.schemas import (
    AnnotationTableResponse,
    ArtifactManifest,
    CompartmentStats,
    ConnectivityResponse,
    CypherQueryResponse,
    EditHistoryResponse,
    NeuronInfoResponse,
    NeuronsByTypeResponse,
    NeuroglancerUrlResponse,
    ProofreadingStatusResponse,
    RegionConnectivityResponse,
    RootIdValidationResponse,
    RootIdValidationResult,
    SynapseCompartmentResponse,
    SynapticPartnerSample,
)


class TestArtifactManifest:
    def test_instantiation(self):
        m = ArtifactManifest(
            artifact_path="/tmp/test.parquet",
            n_rows=100,
            columns=["partner_id", "direction", "n_synapses"],
            schema_description="Columns:\n  partner_id: int64",
            dataset="minnie65",
            query_timestamp="2026-03-10T14:00:00",
        )
        data = m.model_dump()
        assert data["n_rows"] == 100
        assert data["cache_hit"] is False
        assert data["artifact_format"] == "parquet"


class TestNeuronInfoResponse:
    def test_instantiation(self):
        resp = NeuronInfoResponse(
            neuron_id=720575940621039145,
            dataset="minnie65",
            cell_type="L2/3 IT",
            cell_class="excitatory",
            region="VISp",
            soma_position_nm=(200000.0, 300000.0, 400000.0),
            n_pre_synapses=1500,
            n_post_synapses=3200,
            proofread=True,
            materialization_version=943,
            neuroglancer_url="https://neuroglancer.brain-map.org/#!...",
        )
        data = resp.model_dump()
        assert data["neuron_id"] == 720575940621039145
        assert data["cell_type"] == "L2/3 IT"
        assert data["soma_position_nm"] == (200000.0, 300000.0, 400000.0)

    def test_minimal(self):
        resp = NeuronInfoResponse(neuron_id=123, dataset="hemibrain")
        assert resp.warnings == []


class TestConnectivityResponse:
    def test_instantiation(self):
        sample = SynapticPartnerSample(
            partner_id=999,
            partner_type="L4 IT",
            n_synapses=42,
            weight_normalized=0.15,
        )
        manifest = ArtifactManifest(
            artifact_path="/tmp/connectivity.parquet",
            n_rows=200,
            columns=["partner_id", "direction", "n_synapses"],
            schema_description="test",
            dataset="minnie65",
            query_timestamp="2026-03-10T14:00:00",
        )
        resp = ConnectivityResponse(
            neuron_id=123,
            dataset="minnie65",
            n_upstream_total=120,
            n_downstream_total=80,
            upstream_weight_distribution={"mean": 3.5, "median": 2, "max": 50, "p90": 10},
            downstream_weight_distribution={"mean": 4.0, "median": 3, "max": 40, "p90": 12},
            upstream_sample=[sample],
            downstream_sample=[sample],
            neuroglancer_url="https://example.com",
            artifact_manifest=manifest,
        )
        data = resp.model_dump()
        assert data["n_upstream_total"] == 120
        assert len(data["upstream_sample"]) == 1
        assert "sample_note" in data
        assert data["artifact_manifest"]["n_rows"] == 200


class TestNeuronsByTypeResponse:
    def test_instantiation(self):
        resp = NeuronsByTypeResponse(
            dataset="hemibrain",
            query_cell_type="MBON14",
            n_total=5,
            type_distribution={"MBON14": 5},
            region_distribution={"MB(+)": 5},
        )
        assert resp.model_dump()["n_total"] == 5


class TestRegionConnectivityResponse:
    def test_instantiation(self):
        resp = RegionConnectivityResponse(
            dataset="hemibrain",
            n_regions=2,
            top_5_connections=[
                {"source_region": "MB", "target_region": "LH", "n_synapses": 1000}
            ],
            total_synapses=5000,
        )
        data = resp.model_dump()
        assert data["n_regions"] == 2
        assert len(data["top_5_connections"]) == 1


class TestNeuroglancerUrlResponse:
    def test_instantiation(self):
        resp = NeuroglancerUrlResponse(
            url="https://neuroglancer.brain-map.org/#!...",
            dataset="minnie65",
            n_segments=3,
            layers_included=["imagery", "segmentation"],
        )
        assert resp.model_dump()["coordinate_space"] == "nm"


class TestRootIdValidationResponse:
    def test_instantiation(self):
        result = RootIdValidationResult(
            root_id=720575940621039145,
            is_current=False,
            suggested_current_id=720575940621039200,
        )
        resp = RootIdValidationResponse(
            dataset="minnie65",
            materialization_version=943,
            results=[result],
            n_stale=1,
        )
        assert resp.model_dump()["n_stale"] == 1


class TestProofreadingStatusResponse:
    def test_instantiation(self):
        resp = ProofreadingStatusResponse(
            neuron_id=720575940621039145,
            dataset="minnie65",
            axon_proofread=True,
            dendrite_proofread=False,
            strategy_axon="axon_fully_extended",
            n_edits=12,
        )
        assert resp.model_dump()["axon_proofread"] is True


class TestAnnotationTableResponse:
    def test_instantiation(self):
        resp = AnnotationTableResponse(
            dataset="minnie65",
            table_name="nucleus_detection_v0",
            n_total=500,
            schema_description="Columns:\n  id: int64\n  pt_root_id: int64",
        )
        assert resp.model_dump()["n_total"] == 500


class TestEditHistoryResponse:
    def test_instantiation(self):
        resp = EditHistoryResponse(
            neuron_id=720575940621039145,
            dataset="minnie65",
            n_edits_total=5,
            first_edit_timestamp="2024-01-01T00:00:00",
            last_edit_timestamp="2024-06-15T12:00:00",
        )
        assert resp.model_dump()["n_edits_total"] == 5


class TestCypherQueryResponse:
    def test_instantiation(self):
        resp = CypherQueryResponse(
            dataset="hemibrain",
            query="MATCH (n:Neuron) RETURN n LIMIT 10",
            n_rows=10,
            columns=["n"],
        )
        assert resp.model_dump()["n_rows"] == 10


class TestSynapseCompartmentResponse:
    def test_instantiation(self):
        stat = CompartmentStats(compartment="axon", n_synapses=500, fraction=0.65)
        resp = SynapseCompartmentResponse(
            neuron_id=123,
            dataset="hemibrain",
            direction="output",
            compartments=[stat],
            n_total_synapses=770,
        )
        assert resp.model_dump()["direction"] == "output"

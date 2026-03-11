"""neuPrint backend adapter for hemibrain dataset."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from connectomics_mcp.backends.base import ConnectomeBackend
from connectomics_mcp.exceptions import BackendConnectionError

logger = logging.getLogger(__name__)


class NeuPrintBackend(ConnectomeBackend):
    """Backend adapter for neuPrint-based connectomic datasets.

    Parameters
    ----------
    server : str
        neuPrint server hostname.
    dataset : str
        Dataset identifier (e.g. "hemibrain:v1.2.1").
    dataset_name : str
        Human-readable dataset name for error messages.
    """

    def __init__(self, server: str, dataset: str, dataset_name: str) -> None:
        self.server = server
        self.dataset = dataset
        self.dataset_name = dataset_name
        self._client = None

    @property
    def client(self):
        """Lazily initialize the neuPrint client."""
        if self._client is None:
            try:
                from neuprint import Client

                token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "")
                self._client = Client(
                    self.server, dataset=self.dataset, token=token
                )
                logger.debug(
                    "Initialized neuPrint client for %s on %s",
                    self.dataset,
                    self.server,
                )
            except Exception as e:
                raise BackendConnectionError("neuprint", str(e)) from e
        return self._client

    def get_neuron_info(self, neuron_id: int | str) -> dict[str, Any]:
        """Fetch neuron info from neuPrint.

        Parameters
        ----------
        neuron_id : int | str
            Body ID of the neuron.

        Returns
        -------
        dict
            Raw neuron info dict for the formatter.
        """
        body_id = int(neuron_id)
        logger.debug("get_neuron_info(%d) on %s", body_id, self.dataset_name)

        from neuprint import NeuronCriteria as NC, fetch_neurons

        neuron_df, roi_df = fetch_neurons(NC(bodyId=body_id))

        if neuron_df.empty:
            return {
                "neuron_id": body_id,
                "dataset": self.dataset_name,
                "cell_type": None,
                "cell_class": None,
                "region": None,
                "soma_position_nm": None,
                "n_pre_synapses": None,
                "n_post_synapses": None,
                "warnings": [f"No neuron found with bodyId {body_id}"],
            }

        row = neuron_df.iloc[0]

        # Extract soma position if available
        soma_position = None
        soma_loc = row.get("somaLocation", None)
        if soma_loc is not None:
            try:
                coords = soma_loc.get("coordinates", soma_loc)
                if isinstance(coords, (list, tuple)) and len(coords) == 3:
                    soma_position = tuple(float(x) for x in coords)
            except (AttributeError, TypeError):
                pass

        # Primary ROI: pick the ROI with the most post-synapses
        region = None
        if not roi_df.empty:
            roi_for_body = roi_df[roi_df["bodyId"] == body_id]
            if not roi_for_body.empty and "roi" in roi_for_body.columns:
                region = roi_for_body.sort_values("post", ascending=False).iloc[0][
                    "roi"
                ]

        return {
            "neuron_id": body_id,
            "dataset": self.dataset_name,
            "cell_type": row.get("type", None),
            "cell_class": row.get("instance", None),
            "region": region,
            "soma_position_nm": soma_position,
            "n_pre_synapses": int(row["pre"]) if "pre" in row.index else None,
            "n_post_synapses": int(row["post"]) if "post" in row.index else None,
            "warnings": [],
        }

    def get_connectivity(
        self, neuron_id: int | str, direction: str = "both"
    ) -> dict[str, Any]:
        """Fetch all connectivity partners from neuPrint.

        Parameters
        ----------
        neuron_id : int | str
            Body ID of the neuron.
        direction : str
            "upstream", "downstream", or "both".

        Returns
        -------
        dict
            Keys: neuron_id, dataset, warnings, partners_df.
        """
        body_id = int(neuron_id)
        logger.debug("get_connectivity(%d, %s) on %s", body_id, direction, self.dataset_name)

        from neuprint import NeuronCriteria as NC, fetch_adjacencies

        warnings: list[str] = []
        rows: list[dict] = []

        # Upstream: who connects TO this neuron
        if direction in ("both", "upstream"):
            try:
                adj_df, neuron_df = fetch_adjacencies(
                    NC(), NC(bodyId=body_id)
                )
                if not adj_df.empty:
                    # adj_df has columns: bodyId_pre, bodyId_post, weight
                    upstream_counts = adj_df.groupby("bodyId_pre")["weight"].sum()
                    total_input = int(upstream_counts.sum())
                    # Get type info from neuron_df
                    type_map = {}
                    if not neuron_df.empty and "type" in neuron_df.columns:
                        type_map = dict(zip(neuron_df["bodyId"], neuron_df["type"]))
                    for partner_id, n_syn in upstream_counts.items():
                        rows.append({
                            "partner_id": int(partner_id),
                            "direction": "upstream",
                            "n_synapses": int(n_syn),
                            "weight_normalized": float(n_syn) / total_input if total_input > 0 else 0.0,
                            "partner_type": type_map.get(int(partner_id)),
                        })
            except Exception as e:
                logger.warning("Failed to query upstream adjacencies: %s", e)
                warnings.append(f"Failed to query upstream adjacencies: {e}")

        # Downstream: who this neuron connects TO
        if direction in ("both", "downstream"):
            try:
                adj_df, neuron_df = fetch_adjacencies(
                    NC(bodyId=body_id), NC()
                )
                if not adj_df.empty:
                    downstream_counts = adj_df.groupby("bodyId_post")["weight"].sum()
                    total_output = int(downstream_counts.sum())
                    type_map = {}
                    if not neuron_df.empty and "type" in neuron_df.columns:
                        type_map = dict(zip(neuron_df["bodyId"], neuron_df["type"]))
                    for partner_id, n_syn in downstream_counts.items():
                        rows.append({
                            "partner_id": int(partner_id),
                            "direction": "downstream",
                            "n_synapses": int(n_syn),
                            "weight_normalized": float(n_syn) / total_output if total_output > 0 else 0.0,
                            "partner_type": type_map.get(int(partner_id)),
                        })
            except Exception as e:
                logger.warning("Failed to query downstream adjacencies: %s", e)
                warnings.append(f"Failed to query downstream adjacencies: {e}")

        partners_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["partner_id", "direction", "n_synapses", "weight_normalized", "partner_type"]
        )

        # Add placeholder columns expected by the artifact schema
        if not partners_df.empty:
            if "partner_class" not in partners_df.columns:
                partners_df["partner_class"] = None
            if "partner_region" not in partners_df.columns:
                partners_df["partner_region"] = None
            if "neuroglancer_url" not in partners_df.columns:
                partners_df["neuroglancer_url"] = ""

        return {
            "neuron_id": body_id,
            "dataset": self.dataset_name,
            "warnings": warnings,
            "partners_df": partners_df,
        }

    def validate_root_ids(self, root_ids: list[int]) -> dict[str, Any]:
        """Validate neuPrint body IDs — always current (immutable).

        Parameters
        ----------
        root_ids : list[int]
            Body IDs to validate.

        Returns
        -------
        dict
            All IDs marked as current with no timestamps.
        """
        logger.debug("validate_root_ids(%s) on %s", root_ids, self.dataset_name)

        results = [
            {
                "root_id": int(rid),
                "is_current": True,
                "last_edit_timestamp": None,
                "suggested_current_id": None,
            }
            for rid in root_ids
        ]

        return {
            "dataset": self.dataset_name,
            "materialization_version": None,
            "results": results,
            "warnings": [],
        }

    def get_proofreading_status(self, neuron_id: int) -> dict[str, Any]:
        """Not applicable for neuPrint datasets.

        The capability check in the tool layer prevents this from
        being called, but we implement it to satisfy the abstract
        interface.
        """
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported(self.dataset_name, "cave")

    def query_annotation_table(
        self,
        table_name: str,
        filter_equal_dict: dict[str, Any] | None = None,
        filter_in_dict: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Not applicable for neuPrint datasets."""
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported(self.dataset_name, "cave")

    def get_edit_history(self, neuron_id: int) -> dict[str, Any]:
        """Not applicable for neuPrint datasets."""
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported(self.dataset_name, "cave")

    def fetch_cypher(self, query: str) -> dict[str, Any]:
        """Execute a Cypher query against neuPrint.

        Parameters
        ----------
        query : str
            Cypher query string.

        Returns
        -------
        dict
            Keys: dataset, query, materialization_version, warnings, result_df.
        """
        logger.debug("fetch_cypher on %s", self.dataset_name)

        warnings: list[str] = []

        try:
            result_df = self.client.fetch_custom(query)
        except Exception as e:
            logger.warning("Cypher query failed: %s", e)
            warnings.append(f"Cypher query failed: {e}")
            result_df = pd.DataFrame()

        return {
            "dataset": self.dataset_name,
            "query": query,
            "materialization_version": None,
            "warnings": warnings,
            "result_df": result_df,
        }

    def get_synapse_compartments(
        self, neuron_id: int | str, direction: str = "input"
    ) -> dict[str, Any]:
        """Fetch per-ROI synapse distribution from neuPrint.

        Parameters
        ----------
        neuron_id : int | str
            Body ID of the neuron.
        direction : str
            "input" for post-synaptic or "output" for pre-synaptic.

        Returns
        -------
        dict
            Keys: neuron_id, dataset, direction, compartments, n_total_synapses,
            warnings.
        """
        body_id = int(neuron_id)
        logger.debug(
            "get_synapse_compartments(%d, %s) on %s",
            body_id, direction, self.dataset_name,
        )

        from neuprint import NeuronCriteria as NC, fetch_neurons

        warnings: list[str] = []
        compartments: list[dict[str, Any]] = []
        n_total = 0

        try:
            neuron_df, roi_df = fetch_neurons(NC(bodyId=body_id))
        except Exception as e:
            logger.warning("Failed to fetch neuron for compartments: %s", e)
            warnings.append(f"Failed to fetch neuron: {e}")
            return {
                "neuron_id": body_id,
                "dataset": self.dataset_name,
                "direction": direction,
                "compartments": [],
                "n_total_synapses": 0,
                "warnings": warnings,
            }

        if roi_df.empty:
            warnings.append(f"No ROI data found for bodyId {body_id}")
            return {
                "neuron_id": body_id,
                "dataset": self.dataset_name,
                "direction": direction,
                "compartments": [],
                "n_total_synapses": 0,
                "warnings": warnings,
            }

        # Filter to this body's ROI entries
        body_rois = roi_df[roi_df["bodyId"] == body_id]
        if body_rois.empty:
            warnings.append(f"No ROI data found for bodyId {body_id}")
            return {
                "neuron_id": body_id,
                "dataset": self.dataset_name,
                "direction": direction,
                "compartments": [],
                "n_total_synapses": 0,
                "warnings": warnings,
            }

        # Select synapse count column based on direction
        count_col = "post" if direction == "input" else "pre"

        # Build compartment entries from ROI data
        for _, row in body_rois.iterrows():
            roi_name = row.get("roi", "unknown")
            count = int(row.get(count_col, 0))
            if count > 0:
                compartments.append({
                    "compartment": roi_name,
                    "n_synapses": count,
                })
                n_total += count

        # Compute fractions and sort by count descending
        for comp in compartments:
            comp["fraction"] = (
                round(comp["n_synapses"] / n_total, 4) if n_total > 0 else 0.0
            )

        compartments.sort(key=lambda c: c["n_synapses"], reverse=True)

        return {
            "neuron_id": body_id,
            "dataset": self.dataset_name,
            "direction": direction,
            "compartments": compartments,
            "n_total_synapses": n_total,
            "warnings": warnings,
        }

    def get_region_connectivity(
        self,
        source_region: str | None = None,
        target_region: str | None = None,
    ) -> dict[str, Any]:
        """Fetch region-to-region connectivity from neuPrint.

        Parameters
        ----------
        source_region : str, optional
            Filter to connections from this ROI.
        target_region : str, optional
            Filter to connections to this ROI.

        Returns
        -------
        dict
            Keys: dataset, materialization_version, warnings, region_df.
        """
        logger.debug(
            "get_region_connectivity(source=%s, target=%s) on %s",
            source_region, target_region, self.dataset_name,
        )

        from neuprint import fetch_roi_connectivity

        warnings: list[str] = []

        empty_df = pd.DataFrame(
            columns=["source_region", "target_region", "n_synapses",
                     "n_neurons_pre", "n_neurons_post"]
        )

        try:
            roi_conn_df = fetch_roi_connectivity()
        except Exception as e:
            logger.warning("Failed to fetch ROI connectivity: %s", e)
            warnings.append(f"Failed to fetch ROI connectivity: {e}")
            return {
                "dataset": self.dataset_name,
                "materialization_version": None,
                "warnings": warnings,
                "region_df": empty_df,
            }

        if roi_conn_df.empty:
            return {
                "dataset": self.dataset_name,
                "materialization_version": None,
                "warnings": warnings,
                "region_df": empty_df,
            }

        # Map neuPrint ROI connectivity columns to artifact schema
        region_df = pd.DataFrame({
            "source_region": roi_conn_df["from_roi"],
            "target_region": roi_conn_df["to_roi"],
            "n_synapses": roi_conn_df["weight"].astype(int),
            "n_neurons_pre": roi_conn_df.get(
                "upstream_neuron_count",
                pd.Series([0] * len(roi_conn_df)),
            ).astype(int),
            "n_neurons_post": roi_conn_df.get(
                "downstream_neuron_count",
                pd.Series([0] * len(roi_conn_df)),
            ).astype(int),
        })

        # Apply filters
        if source_region:
            region_df = region_df[
                region_df["source_region"].str.contains(
                    source_region, case=False, na=False
                )
            ]
        if target_region:
            region_df = region_df[
                region_df["target_region"].str.contains(
                    target_region, case=False, na=False
                )
            ]

        region_df = region_df.reset_index(drop=True)

        return {
            "dataset": self.dataset_name,
            "materialization_version": None,
            "warnings": warnings,
            "region_df": region_df,
        }

    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        """Fetch neurons matching a cell type from neuPrint.

        Parameters
        ----------
        cell_type : str
            Cell type annotation to search for.
        region : str, optional
            ROI filter.

        Returns
        -------
        dict
            Keys: dataset, query_cell_type, query_region,
            materialization_version, warnings, neurons_df.
        """
        logger.debug(
            "get_neurons_by_type(%s, region=%s) on %s",
            cell_type, region, self.dataset_name,
        )

        from neuprint import NeuronCriteria as NC, fetch_neurons

        warnings: list[str] = []

        criteria_kwargs: dict[str, Any] = {"type": cell_type}
        if region:
            criteria_kwargs["rois"] = [region]

        try:
            neuron_df, roi_df = fetch_neurons(NC(**criteria_kwargs))
        except Exception as e:
            logger.warning("Failed to fetch neurons by type: %s", e)
            warnings.append(f"Failed to fetch neurons: {e}")
            neuron_df = pd.DataFrame()
            roi_df = pd.DataFrame()

        if neuron_df.empty:
            return {
                "dataset": self.dataset_name,
                "query_cell_type": cell_type,
                "query_region": region,
                "materialization_version": None,
                "warnings": warnings,
                "neurons_df": pd.DataFrame(
                    columns=["neuron_id", "cell_type", "cell_class",
                             "region", "n_pre_synapses", "n_post_synapses",
                             "proofread"]
                ),
            }

        # Determine primary ROI per neuron
        region_map: dict[int, str] = {}
        if not roi_df.empty and "roi" in roi_df.columns:
            for body_id in neuron_df["bodyId"].unique():
                body_rois = roi_df[roi_df["bodyId"] == body_id]
                if not body_rois.empty:
                    region_map[int(body_id)] = (
                        body_rois.sort_values("post", ascending=False)
                        .iloc[0]["roi"]
                    )

        rows: list[dict] = []
        for _, row in neuron_df.iterrows():
            body_id = int(row["bodyId"])
            rows.append({
                "neuron_id": body_id,
                "cell_type": row.get("type"),
                "cell_class": row.get("instance"),
                "region": region_map.get(body_id),
                "n_pre_synapses": (
                    int(row["pre"]) if "pre" in row.index and pd.notna(row["pre"]) else None
                ),
                "n_post_synapses": (
                    int(row["post"]) if "post" in row.index and pd.notna(row["post"]) else None
                ),
                "proofread": None,
            })

        neurons_df = pd.DataFrame(rows)

        return {
            "dataset": self.dataset_name,
            "query_cell_type": cell_type,
            "query_region": region,
            "materialization_version": None,
            "warnings": warnings,
            "neurons_df": neurons_df,
        }

"""Abstract base class for connectome backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConnectomeBackend(ABC):
    """Abstract interface for connectome dataset backends.

    Each backend adapts a specific API (CAVE, neuPrint) to the
    universal tool interface. Methods return raw dicts; formatting
    to Pydantic schemas happens in ``output_contracts.formatters``.
    """

    @abstractmethod
    def get_neuron_info(self, neuron_id: int | str) -> dict[str, Any]:
        """Fetch basic neuron information.

        Parameters
        ----------
        neuron_id : int | str
            Root ID (CAVE) or body ID (neuPrint).

        Returns
        -------
        dict
            Raw neuron info dict for the formatter.
        """

    @abstractmethod
    def get_connectivity(
        self, neuron_id: int | str, direction: str = "both"
    ) -> dict[str, Any]:
        """Fetch all connectivity partners.

        Parameters
        ----------
        neuron_id : int | str
            Root ID (CAVE) or body ID (neuPrint).
        direction : str
            "upstream", "downstream", or "both".

        Returns
        -------
        dict
            Raw connectivity dict with ``partners_df`` key for the formatter.
        """

    @abstractmethod
    def search_cell_types(self, query: str) -> dict[str, Any]:
        """Search for cell types matching a query string.

        Performs case-insensitive substring matching across all
        available annotation levels.

        Parameters
        ----------
        query : str
            Search string (e.g. "EPG", "ring", "compass").

        Returns
        -------
        dict
            Keys: dataset, query, matches (list of dicts with
            cell_type, classification_level, n_neurons).
        """

    @abstractmethod
    def get_cell_type_taxonomy(self) -> dict[str, Any]:
        """Return the cell type taxonomy/hierarchy for this dataset.

        Shows the classification levels, top values at each level
        with neuron counts, and example lineages (full path from
        broadest to finest level).

        Returns
        -------
        dict
            Keys: dataset, n_total_neurons, levels (list of level
            dicts), example_lineages (list of path dicts).
        """

    @abstractmethod
    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        """Fetch neurons matching a cell type.

        Parameters
        ----------
        cell_type : str
            Cell type annotation to search for.
        region : str, optional
            Brain region filter.

        Returns
        -------
        dict
            Raw neuron list dict for the formatter.
        """

    @abstractmethod
    def validate_root_ids(self, root_ids: list[int]) -> dict[str, Any]:
        """Check whether root IDs are current.

        Parameters
        ----------
        root_ids : list[int]
            Root IDs (CAVE) or body IDs (neuPrint) to validate.

        Returns
        -------
        dict
            Raw validation results for the formatter.
        """

    @abstractmethod
    def get_region_connectivity(
        self,
        source_region: str | None = None,
        target_region: str | None = None,
    ) -> dict[str, Any]:
        """Fetch region-to-region connectivity.

        Parameters
        ----------
        source_region : str, optional
            Filter to connections from this region.
        target_region : str, optional
            Filter to connections to this region.

        Returns
        -------
        dict
            Raw region connectivity dict with ``region_df`` key for the formatter.
        """

    @abstractmethod
    def get_proofreading_status(self, neuron_id: int) -> dict[str, Any]:
        """Fetch proofreading status for a neuron.

        Parameters
        ----------
        neuron_id : int
            Root ID of the neuron.

        Returns
        -------
        dict
            Raw proofreading status for the formatter.
        """

    @abstractmethod
    def query_annotation_table(
        self,
        table_name: str,
        filter_equal_dict: dict[str, Any] | None = None,
        filter_in_dict: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Query an annotation table.

        Parameters
        ----------
        table_name : str
            Name of the annotation table to query.
        filter_equal_dict : dict, optional
            Equality filters passed to the query.
        filter_in_dict : dict, optional
            Membership filters passed to the query.

        Returns
        -------
        dict
            Raw annotation table dict with ``table_df`` key for the formatter.
        """

    @abstractmethod
    def get_edit_history(self, neuron_id: int) -> dict[str, Any]:
        """Fetch edit history for a neuron.

        Parameters
        ----------
        neuron_id : int
            Root ID of the neuron.

        Returns
        -------
        dict
            Raw edit history dict with ``edits_df`` key for the formatter.
        """

    @abstractmethod
    def fetch_cypher(self, query: str) -> dict[str, Any]:
        """Execute a Cypher query against the backend.

        Parameters
        ----------
        query : str
            Cypher query string.

        Returns
        -------
        dict
            Raw query result dict with ``result_df`` key for the formatter.
        """

    @abstractmethod
    def get_bulk_connectivity(
        self, root_ids: list[int], direction: str = "both"
    ) -> dict[str, Any]:
        """Fetch connectivity for multiple neurons in bulk.

        Parameters
        ----------
        root_ids : list[int]
            Root IDs (CAVE) or body IDs (neuPrint) to query.
        direction : str
            "pre" (outgoing), "post" (incoming), or "both".

        Returns
        -------
        dict
            Keys: edges_df (columns: pre_root_id, post_root_id,
            syn_count, neuropil), materialization_version, warnings.
        """

    @abstractmethod
    def get_synapse_compartments(
        self, neuron_id: int | str, direction: str = "input"
    ) -> dict[str, Any]:
        """Fetch synapse distribution across compartments/ROIs.

        Parameters
        ----------
        neuron_id : int | str
            Body ID of the neuron.
        direction : str
            "input" for post-synaptic or "output" for pre-synaptic.

        Returns
        -------
        dict
            Raw compartment stats for the formatter.
        """

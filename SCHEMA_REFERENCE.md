# SCHEMA_REFERENCE.md — Verified Column Names Per Table Per Dataset

Last verified: 2026-03-10 via `tests/integration/test_schema_audit.py`

## minnie65 (CAVE — `minnie65_public`)

### `aibs_metamodel_celltypes_v661` (cell type table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | |
| `classification_system` | string | e.g. "excitatory_neuron" |
| `cell_type` | string | e.g. "5P-IT" |
| `volume` | Float32 | |
| `pt_supervoxel_id` | Int64 | |
| `pt_root_id` | Int64 | **Primary key for joins** |
| `pt_position` | object (array) | [x, y, z] |
| `bb_start_position` | object | |
| `bb_end_position` | object | |

### `synapses_pni_2` (synapse table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | |
| `pre_pt_root_id` | Int64 | **Presynaptic neuron** |
| `post_pt_root_id` | Int64 | **Postsynaptic neuron** |
| `size` | Float32 | |
| `pre_pt_position` | object | |
| `post_pt_position` | object | |
| `ctr_pt_position` | object | |

### `nucleus_detection_v0` (nucleus table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | **Nucleus ID** |
| `pt_root_id` | Int64 | **Maps to root ID** |
| `volume` | Float32 | |
| `pt_position` | object (array) | |

### `proofreading_status_and_strategy` (proofreading table)
| Column | Type | Notes |
|--------|------|-------|
| `pt_root_id` | Int64 | |
| `status_dendrite` | boolean | True/False |
| `status_axon` | boolean | True/False |
| `strategy_dendrite` | string | e.g. "dendrite_extended" |
| `strategy_axon` | string | e.g. "axon_fully_extended" |
| `valid_id` | Int64 | |

### `chunkedgraph.get_tabular_change_log([root_id])` (changelog)
- **Return type**: `dict[int, pd.DataFrame]` — keyed by root_id
- **Columns**:

| Column | Type | Notes |
|--------|------|-------|
| `operation_id` | int64 | |
| `timestamp` | int64 | **Milliseconds since epoch** |
| `user_id` | str | |
| `is_merge` | bool | True=merge, False=split |
| `before_root_ids` | object (list) | |
| `after_root_ids` | object (list) | |
| `user_name` | str | |
| `user_affiliation` | str | |

---

## flywire (CAVE — `flywire_fafb_public`)

Available tables (6): `fly_synapses_neuropil_v6`, `hierarchical_neuron_annotations`, `neuron_information_v2`, `nuclei_v1`, `proofread_neurons`, `synapses_nt_v1`

### `hierarchical_neuron_annotations` (hierarchical classification)
| Column | Type | Notes |
|--------|------|-------|
| `pt_root_id` | Int64 | |
| `classification_system` | string | Level: `super_class`, `cell_class`, `cell_sub_class`, `cell_type` |
| `cell_type` | string | Label at that classification level |

**Note**: Cannot be filtered server-side by `pt_root_id` (returns 500 KeyError). Must fetch full table and filter in Python. Cached in memory with 10-minute TTL.

### `neuron_information_v2` (cell type table — free-form tags)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | |
| `tag` | string | **Cell type label** (not `cell_type`) |
| `user_id` | Int32 | |
| `pt_root_id` | Int64 | |
| `pt_position` | object | |

### `synapses_nt_v1` (synapse table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | |
| `pre_pt_root_id` | Int64 | |
| `post_pt_root_id` | Int64 | |
| `connection_score` | Float32 | |
| `cleft_score` | Float32 | |
| `gaba`, `ach`, `glut`, `oct`, `ser`, `da` | Float32 | Neurotransmitter predictions |
| `valid_nt` | boolean | |

**Note**: `select_columns=["id"]` fails with server error. Use `select_columns=["pre_pt_root_id"]` or `["post_pt_root_id"]` instead for synapse counts.

### `nuclei_v1` (nucleus table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | |
| `pt_root_id` | Int64 | |
| `pt_position` | object | |

### `proofread_neurons` (proofreading table)
| Column | Type | Notes |
|--------|------|-------|
| `id` | Int64 | |
| `pt_root_id` | Int64 | |
| `pt_position` | object | |

**Note**: No `status_axon`/`status_dendrite` columns. Presence in this table = neuron is proofread.

---

## FANC (CAVE — `fanc_production_mar2021`)

Access requires specific permissions; not verified in this audit. Table names assumed:
- Cell type: `cell_info`
- Synapses: `synapses`
- Nucleus: `nuclei_v1`
- Proofreading: `proofreading_status`

---

## hemibrain (neuPrint — `hemibrain:v1.2.1`)

### `fetch_neurons()` → `(neuron_df, roi_df)`

**neuron_df columns**:
| Column | Type | Notes |
|--------|------|-------|
| `bodyId` | int64 | **Primary key** |
| `instance` | str | e.g. "DA1_adPN_R" |
| `type` | str | **Cell type** |
| `pre` | int64 | Presynaptic count |
| `post` | int64 | Postsynaptic count |
| `downstream` | int64 | |
| `upstream` | int64 | |
| `somaLocation` | object | **`list` [x, y, z]**, NOT dict |
| `somaRadius` | int64 | |
| `status` | str | |
| `cropped` | bool | |
| `inputRois` | object (list) | |
| `outputRois` | object (list) | |

**roi_df columns**:
| Column | Type | Notes |
|--------|------|-------|
| `bodyId` | int64 | |
| `roi` | str | ROI name |
| `pre` | int64 | Pre count in this ROI |
| `post` | int64 | Post count in this ROI |
| `downstream` | int64 | |
| `upstream` | int64 | |

### `fetch_adjacencies()` → `(neuron_df, conn_df)`

**conn_df columns**:
| Column | Type | Notes |
|--------|------|-------|
| `bodyId_pre` | int64 | |
| `bodyId_post` | int64 | |
| `roi` | str | |
| `weight` | int64 | Synapse count in that ROI |

### ROI Connectivity (via Cypher)

`fetch_roi_connectivity` does **not exist** in neuprint-python 0.6. Use Cypher query instead.

**Result columns**:
| Column | Type | Notes |
|--------|------|-------|
| `from_roi` | str | Source ROI |
| `to_roi` | str | Target ROI |
| `n_synapses` | int64 | Total postsynaptic count |
| `n_connections` | int64 | Number of neuron-neuron connections |

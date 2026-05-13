"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        target_hist_match_config: Optional[Dict[str, Any]] = None,
        recent_activity_config: Optional[Dict[str, Any]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            target_hist_match_config: optional configuration for dynamic
                target-history matching item-int features.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.target_hist_match_config = target_hist_match_config or {}
        self.use_target_hist_match = bool(
            self.target_hist_match_config.get('enabled', False)
        )
        self.recent_activity_config = recent_activity_config or {}
        self.use_recent_activity = bool(
            self.recent_activity_config.get('enabled', False)
        )
        self.recent_activity_mode = str(
            self.recent_activity_config.get('mode', 'per_domain')
        )
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}
        self._seq_fid_to_slot: Dict[str, Dict[int, int]] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self._seq_fid_to_slot[domain] = {
                fid: i for i, fid in enumerate(sideinfo)
            }
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

        self.target_hist_match_feature_ids: List[int] = []
        self._target_hist_match_offset = self.item_int_schema.total_dim
        if self.use_target_hist_match:
            self._init_target_hist_match_schema()

        self.recent_activity_feature_ids: List[int] = []
        self._recent_activity_offset = self.user_int_schema.total_dim
        if self.use_recent_activity:
            self._init_recent_activity_schema()

    def _init_target_hist_match_schema(self) -> None:
        """Append dynamic target-history matching features to item_int schema."""
        required = [
            'target_cate_item_fid',
            'hist_cate_seq_fids',
            'feature_fids',
        ]
        missing = [k for k in required if k not in self.target_hist_match_config]
        if missing:
            raise ValueError(
                f"target_hist_match_config missing required keys: {missing}"
            )

        feature_fids = list(self.target_hist_match_config['feature_fids'])
        timewise_count = 7
        if len(feature_fids) not in (4, timewise_count):
            raise ValueError(
                "target_hist_match_config['feature_fids'] must contain either "
                f"4 fids for target_cate_match_v1 or {timewise_count} fids "
                "for target_cate_state timewise features"
            )

        existing = set(self.item_int_schema.feature_ids)
        dup = [fid for fid in feature_fids if fid in existing]
        if dup:
            raise ValueError(
                f"target_hist_match feature fids already exist in item_int schema: {dup}"
            )

        self.target_hist_match_recent_k = int(
            self.target_hist_match_config.get('recent_k', 64)
        )

        # Max bucket ids for:
        # cate_in_hist, cate_count_bucket, cate_ratio_bucket,
        # cate_last_time_delta_bucket, then optional:
        # cate_last_position_delta_bucket, cate_recent_ratio_bucket,
        # cate_recent_trend_bucket.
        vocab_sizes = [1, 5, 4, 5]
        if len(feature_fids) == timewise_count:
            vocab_sizes += [7, 5, 5]
        for fid, vs in zip(feature_fids, vocab_sizes):
            self.item_int_schema.add(int(fid), 1)
            self.item_int_vocab_sizes.append(vs)
        self.target_hist_match_feature_ids = [int(fid) for fid in feature_fids]
        self._target_hist_match_feature_count = len(feature_fids)
        self._target_hist_match_has_timewise = len(feature_fids) == timewise_count

        target_cate_fid = int(self.target_hist_match_config['target_cate_item_fid'])
        self._target_cate_offset, self._target_cate_dim = (
            self.item_int_schema.get_offset_length(target_cate_fid)
        )

        hist_cate_fids = {
            str(domain): int(fid)
            for domain, fid in self.target_hist_match_config['hist_cate_seq_fids'].items()
        }
        self._target_hist_cate_slots = {}
        for domain, fid in hist_cate_fids.items():
            if domain not in self._seq_fid_to_slot:
                raise ValueError(f"Unknown sequence domain in hist_cate_seq_fids: {domain}")
            if fid not in self._seq_fid_to_slot[domain]:
                raise ValueError(
                    f"hist_cate_seq_fids[{domain}]={fid} not found in that sequence domain"
                )
            self._target_hist_cate_slots[domain] = self._seq_fid_to_slot[domain][fid]

    def _init_recent_activity_schema(self) -> None:
        """Append dynamic recent-activity bucket features to user_int schema."""
        feature_fids = list(self.recent_activity_config.get('feature_fids', []))
        windows_seconds = list(
            self.recent_activity_config.get('windows_seconds', [3600, 86400, 7 * 86400])
        )
        self.recent_activity_mode = str(
            self.recent_activity_config.get('mode', 'per_domain')
        )
        if self.recent_activity_mode not in ('per_domain', 'global'):
            raise ValueError(
                "recent_activity_config['mode'] must be 'per_domain' or 'global'"
            )
        features_per_group = 1 + len(windows_seconds)
        expected = features_per_group
        if self.recent_activity_mode == 'per_domain':
            expected *= len(self.seq_domains)
        if len(feature_fids) != expected:
            raise ValueError(
                "recent_activity_config['feature_fids'] must contain "
                f"{expected} fids for mode={self.recent_activity_mode}: "
                f"one last_delta bucket plus {len(windows_seconds)} recent-count buckets"
                + (
                    f" for each of {len(self.seq_domains)} domains"
                    if self.recent_activity_mode == 'per_domain'
                    else ""
                )
            )

        existing = set(self.user_int_schema.feature_ids)
        dup = [fid for fid in feature_fids if fid in existing]
        if dup:
            raise ValueError(
                f"recent_activity feature fids already exist in user_int schema: {dup}"
            )

        for fid in feature_fids:
            self.user_int_schema.add(int(fid), 1)
            self.user_int_vocab_sizes.append(5)
        self.recent_activity_feature_ids = [int(fid) for fid in feature_fids]
        self.recent_activity_windows_seconds = [int(w) for w in windows_seconds]
        self._recent_activity_features_per_group = features_per_group
        self._recent_activity_features_per_domain = features_per_group
        self._recent_activity_domain_offset = {}
        if self.recent_activity_mode == 'per_domain':
            self._recent_activity_domain_offset = {
                domain: self._recent_activity_offset + i * self._recent_activity_features_per_group
                for i, domain in enumerate(self.seq_domains)
            }

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    @staticmethod
    def _bucket_target_cate_count(count: "npt.NDArray[np.int64]") -> "npt.NDArray[np.int64]":
        bucket = np.zeros_like(count, dtype=np.int64)
        bucket[count == 1] = 1
        bucket[count == 2] = 2
        bucket[(count >= 3) & (count <= 5)] = 3
        bucket[(count >= 6) & (count <= 10)] = 4
        bucket[count > 10] = 5
        return bucket

    @staticmethod
    def _bucket_target_cate_ratio(
        count: "npt.NDArray[np.int64]",
        hist_len: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.int64]":
        bucket = np.zeros_like(count, dtype=np.int64)
        valid = (count > 0) & (hist_len > 0)
        ratio = np.zeros(count.shape, dtype=np.float32)
        ratio[valid] = count[valid] / hist_len[valid].astype(np.float32)
        bucket[(ratio > 0.0) & (ratio <= 0.1)] = 1
        bucket[(ratio > 0.1) & (ratio <= 0.3)] = 2
        bucket[(ratio > 0.3) & (ratio <= 0.5)] = 3
        bucket[ratio > 0.5] = 4
        return bucket

    @staticmethod
    def _bucket_target_last_delta(delta_seconds: "npt.NDArray[np.int64]") -> "npt.NDArray[np.int64]":
        bucket = np.zeros(delta_seconds.shape, dtype=np.int64)
        valid = delta_seconds >= 0
        bucket[valid & (delta_seconds <= 3600)] = 1
        bucket[valid & (delta_seconds > 3600) & (delta_seconds <= 86400)] = 2
        bucket[valid & (delta_seconds > 86400) & (delta_seconds <= 7 * 86400)] = 3
        bucket[valid & (delta_seconds > 7 * 86400) & (delta_seconds <= 30 * 86400)] = 4
        bucket[valid & (delta_seconds > 30 * 86400)] = 5
        return bucket

    @staticmethod
    def _bucket_target_recent_rate(rate: "npt.NDArray[np.float32]") -> "npt.NDArray[np.int64]":
        bucket = np.zeros(rate.shape, dtype=np.int64)
        bucket[(rate > 0.0) & (rate <= 0.05)] = 1
        bucket[(rate > 0.05) & (rate <= 0.10)] = 2
        bucket[(rate > 0.10) & (rate <= 0.20)] = 3
        bucket[(rate > 0.20) & (rate <= 0.40)] = 4
        bucket[rate > 0.40] = 5
        return bucket

    @staticmethod
    def _bucket_target_trend(
        trend: "npt.NDArray[np.float32]",
        hist_len: "npt.NDArray[np.int64]",
    ) -> "npt.NDArray[np.int64]":
        bucket = np.zeros(trend.shape, dtype=np.int64)  # 0 = no_history
        valid = hist_len > 0
        bucket[valid & (trend <= -0.10)] = 1
        bucket[valid & (trend > -0.10) & (trend < -0.03)] = 2
        bucket[valid & (trend >= -0.03) & (trend <= 0.03)] = 3
        bucket[valid & (trend > 0.03) & (trend <= 0.10)] = 4
        bucket[valid & (trend > 0.10)] = 5
        return bucket

    @staticmethod
    def _bucket_target_position_delta(delta: "npt.NDArray[np.int64]") -> "npt.NDArray[np.int64]":
        bucket = np.zeros(delta.shape, dtype=np.int64)  # 0 = never
        valid = delta >= 0
        bucket[valid & (delta <= 1)] = 1
        bucket[valid & (delta >= 2) & (delta <= 5)] = 2
        bucket[valid & (delta >= 6) & (delta <= 10)] = 3
        bucket[valid & (delta >= 11) & (delta <= 30)] = 4
        bucket[valid & (delta >= 31) & (delta <= 80)] = 5
        bucket[valid & (delta >= 81) & (delta <= 160)] = 6
        bucket[valid & (delta > 160)] = 7
        return bucket

    def _append_target_timewise_stats(
        self,
        accum: Dict[str, Any],
        cate_valid: "npt.NDArray[np.bool_]",
        cate_match: "npt.NDArray[np.bool_]",
        seq_timestamps: Optional["npt.NDArray[np.int64]"],
    ) -> None:
        """Append per-domain valid/match timestamps for one-pass finalization."""
        if not self._target_hist_match_has_timewise or seq_timestamps is None:
            return

        start = int(accum['timewise_offset'])
        end = start + seq_timestamps.shape[1]
        valid_ts = np.where(cate_valid, seq_timestamps, 0)
        accum['timewise_valid_ts'][:, start:end] = valid_ts
        accum['timewise_match'][:, start:end] = cate_match & cate_valid
        accum['timewise_offset'] = end

    def _accumulate_target_hist_match(
        self,
        *,
        domain: str,
        seq_values: "npt.NDArray[np.int64]",
        seq_timestamps: Optional["npt.NDArray[np.int64]"],
        current_timestamps: "npt.NDArray[np.int64]",
        target_cate_values: "npt.NDArray[np.int64]",
        seq_lengths: "npt.NDArray[np.int64]",
        accum: Dict[str, Any],
    ) -> None:
        """Accumulate v1 target-history matching stats from one sequence domain."""
        B, _, L = seq_values.shape
        positions = np.arange(L).reshape(1, L)
        base_valid = positions < seq_lengths.reshape(B, 1)
        if seq_timestamps is not None:
            base_valid &= (seq_timestamps > 0) & (seq_timestamps < current_timestamps.reshape(B, 1))

        cate_slot = self._target_hist_cate_slots.get(domain)
        if cate_slot is None:
            return

        hist_cate = seq_values[:, cate_slot, :]
        cate_valid = base_valid & (hist_cate > 0)
        accum['hist_cate_len'] += cate_valid.sum(axis=1).astype(np.int64)

        target_cate_valid = target_cate_values > 0
        if target_cate_values.shape[1] == 1:
            target_cate = target_cate_values[:, 0].reshape(B, 1)
            cate_match = (
                cate_valid
                & target_cate_valid[:, 0].reshape(B, 1)
                & (hist_cate == target_cate)
            )
        else:
            cate_match = (
                cate_valid[:, :, None]
                & target_cate_valid[:, None, :]
                & (hist_cate[:, :, None] == target_cate_values[:, None, :])
            ).any(axis=2)

        accum['cate_match_count'] += cate_match.sum(axis=1).astype(np.int64)
        if seq_timestamps is not None:
            delta = current_timestamps.reshape(B, 1) - seq_timestamps
            masked_delta = np.where(cate_match, delta, np.iinfo(np.int64).max)
            accum['cate_last_delta'] = np.minimum(
                accum['cate_last_delta'],
                masked_delta.min(axis=1),
            )
        self._append_target_timewise_stats(accum, cate_valid, cate_match, seq_timestamps)

    def _build_target_timewise_features(
        self,
        accum: Dict[str, Any],
    ) -> List["npt.NDArray[np.int64]"]:
        """Build global target-cate timewise buckets from accumulated buffers."""
        valid_ts = accum['timewise_valid_ts']
        match = accum['timewise_match']
        B, total_len = valid_ts.shape
        has_valid = valid_ts > 0
        has_match = match & has_valid

        latest_match_ts = np.where(has_match, valid_ts, 0).max(axis=1)
        position_delta = np.full(B, -1, dtype=np.int64)
        matched = latest_match_ts > 0
        if matched.any():
            position_delta[matched] = (
                valid_ts[matched] > latest_match_ts[matched, None]
            ).sum(axis=1).astype(np.int64)

        recent_k = min(int(self.target_hist_match_recent_k), total_len)
        if recent_k <= 0:
            recent_len = np.zeros(B, dtype=np.int64)
            recent_count = np.zeros(B, dtype=np.int64)
        else:
            order = np.argpartition(-valid_ts, recent_k - 1, axis=1)[:, :recent_k]
            recent_ts = np.take_along_axis(valid_ts, order, axis=1)
            recent_match = np.take_along_axis(match, order, axis=1)
            recent_valid = recent_ts > 0
            recent_len = recent_valid.sum(axis=1).astype(np.int64)
            recent_count = (recent_match & recent_valid).sum(axis=1).astype(np.int64)

        recent_rate = np.zeros(B, dtype=np.float32)
        long_rate = np.zeros(B, dtype=np.float32)
        recent_nonzero = recent_len > 0
        long_nonzero = accum['hist_cate_len'] > 0
        recent_rate[recent_nonzero] = (
            recent_count[recent_nonzero] / recent_len[recent_nonzero].astype(np.float32)
        )
        long_rate[long_nonzero] = (
            accum['cate_match_count'][long_nonzero]
            / accum['hist_cate_len'][long_nonzero].astype(np.float32)
        )
        trend = recent_rate - long_rate

        return [
            self._bucket_target_position_delta(position_delta),
            self._bucket_target_recent_rate(recent_rate),
            self._bucket_target_trend(trend, accum['hist_cate_len']),
        ]

    def _write_target_hist_match_features(
        self,
        item_int: "npt.NDArray[np.int64]",
        accum: Dict[str, Any],
    ) -> None:
        """Write v1 target-history matching buckets into appended item_int columns."""
        if not self.use_target_hist_match:
            return

        cate_count = accum['cate_match_count']
        last_delta = accum['cate_last_delta']
        last_delta = np.where(last_delta == np.iinfo(np.int64).max, -1, last_delta)

        feat_parts = [
            (cate_count > 0).astype(np.int64),
            self._bucket_target_cate_count(cate_count),
            self._bucket_target_cate_ratio(cate_count, accum['hist_cate_len']),
            self._bucket_target_last_delta(last_delta),
        ]
        if self._target_hist_match_has_timewise:
            feat_parts.extend(self._build_target_timewise_features(accum))

        feats = np.stack(feat_parts, axis=1)
        end = self._target_hist_match_offset + self._target_hist_match_feature_count
        item_int[:, self._target_hist_match_offset:end] = feats

    def _write_recent_activity_features(
        self,
        *,
        domain: str,
        user_int: "npt.NDArray[np.int64]",
        seq_timestamps: Optional["npt.NDArray[np.int64]"],
        current_timestamps: "npt.NDArray[np.int64]",
        seq_lengths: "npt.NDArray[np.int64]",
    ) -> None:
        """Write per-domain recent-activity buckets into appended user_int columns."""
        if not self.use_recent_activity or seq_timestamps is None:
            return

        B, L = seq_timestamps.shape
        positions = np.arange(L).reshape(1, L)
        valid = (
            (positions < seq_lengths.reshape(B, 1))
            & (seq_timestamps > 0)
            & (seq_timestamps < current_timestamps.reshape(B, 1))
        )
        delta = current_timestamps.reshape(B, 1) - seq_timestamps

        masked_delta = np.where(valid, delta, np.iinfo(np.int64).max)
        last_delta = masked_delta.min(axis=1)
        last_delta = np.where(last_delta == np.iinfo(np.int64).max, -1, last_delta)

        feats = [self._bucket_target_last_delta(last_delta)]
        for window in self.recent_activity_windows_seconds:
            count = (valid & (delta <= int(window))).sum(axis=1).astype(np.int64)
            feats.append(self._bucket_target_cate_count(count))

        start = self._recent_activity_domain_offset[domain]
        end = start + self._recent_activity_features_per_group
        user_int[:, start:end] = np.stack(feats, axis=1)

    def _write_global_recent_activity_features(
        self,
        *,
        user_int: "npt.NDArray[np.int64]",
        seq_timestamps_list: List["npt.NDArray[np.int64]"],
        seq_lengths_list: List["npt.NDArray[np.int64]"],
        current_timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Write global recent-activity buckets over all sequence domains."""
        if not self.use_recent_activity or not seq_timestamps_list:
            return

        valid_parts = []
        delta_parts = []
        B = current_timestamps.shape[0]
        current_ts = current_timestamps.reshape(B, 1)
        for seq_timestamps, seq_lengths in zip(seq_timestamps_list, seq_lengths_list):
            _, L = seq_timestamps.shape
            positions = np.arange(L).reshape(1, L)
            valid = (
                (positions < seq_lengths.reshape(B, 1))
                & (seq_timestamps > 0)
                & (seq_timestamps < current_ts)
            )
            delta = current_ts - seq_timestamps
            valid_parts.append(valid)
            delta_parts.append(delta)

        valid_all = np.concatenate(valid_parts, axis=1)
        delta_all = np.concatenate(delta_parts, axis=1)

        masked_delta = np.where(valid_all, delta_all, np.iinfo(np.int64).max)
        last_delta = masked_delta.min(axis=1)
        last_delta = np.where(last_delta == np.iinfo(np.int64).max, -1, last_delta)

        feats = [self._bucket_target_last_delta(last_delta)]
        for window in self.recent_activity_windows_seconds:
            count = (valid_all & (delta_all <= int(window))).sum(axis=1).astype(np.int64)
            feats.append(self._bucket_target_cate_count(count))

        start = self._recent_activity_offset
        end = start + self._recent_activity_features_per_group
        user_int[:, start:end] = np.stack(feats, axis=1)

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()
        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        target_match_accum = None
        target_cate_values = None
        if self.use_target_hist_match:
            target_cate_values = item_int[
                :, self._target_cate_offset:self._target_cate_offset + self._target_cate_dim
            ].copy()
            target_match_accum = {
                'cate_match_count': np.zeros(B, dtype=np.int64),
                'hist_cate_len': np.zeros(B, dtype=np.int64),
                'cate_last_delta': np.full(B, np.iinfo(np.int64).max, dtype=np.int64),
            }
            if self._target_hist_match_has_timewise:
                timewise_width = sum(self._seq_maxlen[domain] for domain in self.seq_domains)
                target_match_accum['timewise_valid_ts'] = np.zeros(
                    (B, timewise_width), dtype=np.int64
                )
                target_match_accum['timewise_match'] = np.zeros(
                    (B, timewise_width), dtype=bool
                )
                target_match_accum['timewise_offset'] = 0

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        recent_activity_ts_list: List["npt.NDArray[np.int64]"] = []
        recent_activity_len_list: List["npt.NDArray[np.int64]"] = []
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            ts_padded = None
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            if ts_padded is None:
                result[f'{domain}_timestamp'] = torch.zeros(B, max_len, dtype=torch.long)
            else:
                result[f'{domain}_timestamp'] = torch.from_numpy(ts_padded.copy())

            if (
                self.use_recent_activity
                and self.recent_activity_mode == 'per_domain'
            ):
                self._write_recent_activity_features(
                    domain=domain,
                    user_int=user_int,
                    seq_timestamps=ts_padded,
                    current_timestamps=timestamps,
                    seq_lengths=lengths,
                )
            elif (
                self.use_recent_activity
                and self.recent_activity_mode == 'global'
                and ts_padded is not None
            ):
                recent_activity_ts_list.append(ts_padded)
                recent_activity_len_list.append(lengths.copy())

            if self.use_target_hist_match and target_match_accum is not None:
                self._accumulate_target_hist_match(
                    domain=domain,
                    seq_values=out,
                    seq_timestamps=ts_padded if ts_ci is not None else None,
                    current_timestamps=timestamps,
                    target_cate_values=target_cate_values,
                    seq_lengths=lengths,
                    accum=target_match_accum,
                )

        if (
            self.use_recent_activity
            and self.recent_activity_mode == 'global'
        ):
            self._write_global_recent_activity_features(
                user_int=user_int,
                seq_timestamps_list=recent_activity_ts_list,
                seq_lengths_list=recent_activity_len_list,
                current_timestamps=timestamps,
            )

        if self.use_target_hist_match and target_match_accum is not None:
            self._write_target_hist_match_features(item_int, target_match_accum)
            result['item_int_feats'] = torch.from_numpy(item_int.copy())

        if self.use_recent_activity:
            result['user_int_feats'] = torch.from_numpy(user_int.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    target_hist_match_config: Optional[Dict[str, Any]] = None,
    recent_activity_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Validation is the last ``valid_ratio`` fraction of Row Groups in file
    order. ``train_ratio`` optionally keeps only the first fraction of the
    training Row Groups.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])
    train_row_group_range = (0, n_train_rgs)
    valid_row_group_range = (n_train_rgs, total_rgs)

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        target_hist_match_config=target_hist_match_config,
        recent_activity_config=recent_activity_config,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=train_row_group_range,
        clip_vocab=clip_vocab,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        target_hist_match_config=target_hist_match_config,
        recent_activity_config=recent_activity_config,
        shuffle=False,
        buffer_batches=0,
        row_group_range=valid_row_group_range,
        clip_vocab=clip_vocab,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet Row Group split: "
                 f"train={train_rows} rows, valid={valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset

import pandas as pd
import numpy as np
import torch
import logging
from tqdm import tqdm
from src.data.components.c_preprocessing import AlgorithmProcessor

log = logging.getLogger(__name__)

class HistoryAugmentedProcessor(AlgorithmProcessor):
    """
    History-Augmented Processor

    Pre-computes fixed-length sequences with Left Padding and Type Embeddings.
    Structure: [PADDING | HISTORY | CURRENT]
    """

    def __init__(
        self,
        max_time_gap_in_one_trajectory_hours: int = 24,
        split_ratio_list: list = [0.7, 0.2, 0.1],
        min_trajectory_length: int = 2,
        max_trajectory_length: int = 100,
        iterative_filter_sparse_poi_user: bool = True,
        sequence_length: int = 1000,
        use_history: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.max_time_gap_in_one_trajectory_hours = max_time_gap_in_one_trajectory_hours
        self.split_ratio_list = split_ratio_list
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.iterative_filter_sparse_poi_user = iterative_filter_sparse_poi_user
        self.sequence_length = sequence_length
        self.use_history = use_history

        self.priors = {}
        self.stats = {}
        self.augmented_cache = {}
        self.final_df = None

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        log.info(f"Starting History-Augmented Processing (Max Len={self.sequence_length})...")
        df = self._base_process(data)
        df = self._enrich_features(df)
        self._calculate_global_priors(df)
        df = self._normalize_coords(df)
        self._extract_poi_coords(df)
        self._collect_stats(df)
        self._build_augmented_sequences(df)
        self.final_df = df
        return df

    def _enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["user_id", "location_id", "category_name"]:
            if col in df.columns:
                target_col = "category_id" if col == "category_name" else col
                df[target_col] = df[col].astype("category").cat.codes

        if "abs_time" not in df.columns:
            df["abs_time"] = df["local_time"].astype("int64") // 10**9

        df["weekday"] = df["local_time"].dt.weekday
        df["hour"] = df["local_time"].dt.hour
        df["time_slot"] = df["weekday"] * 24 + df["hour"]

        df = df.sort_values(["user_id", "abs_time"])
        df["delta_time"] = df.groupby("user_id")["abs_time"].diff().fillna(0) / 3600.0
        return df.reset_index(drop=True)

    def _build_augmented_sequences(self, df: pd.DataFrame):
        log.info(f"Building augmented sequences (L={self.sequence_length}, Left-Padding)...")

        df = df.sort_values(["user_id", "local_time"])
        cols = ["trajectory_id", "user_id", "location_id", "category_id",
                "time_slot", "delta_time", "norm_lat", "norm_lon", "abs_time"]

        grand_dict = {}
        grouped = df.groupby("trajectory_id")
        for tid, group in tqdm(grouped, desc="Indexing Trajectories"):
            grand_dict[tid] = {c: group[c].values for c in cols}

        user_traj_df = df[["user_id", "trajectory_id", "local_time"]].drop_duplicates("trajectory_id")
        user_traj_df = user_traj_df.sort_values(["user_id", "local_time"])
        user_traj_map = user_traj_df.groupby("user_id")["trajectory_id"].apply(list).to_dict()

        processed_cache = {}

        PAD_LOC = self.stats["num_locations"]
        PAD_CAT = self.stats["num_categories"]
        PAD_SLOT = 168

        TYPE_PAD = 0
        TYPE_HIST = 1
        TYPE_CURR = 2

        for tid, data_curr in tqdm(grand_dict.items(), desc="Building Sequences"):
            uid = data_curr["user_id"][0]

            # Current trajectory arrays
            c_loc = data_curr["location_id"]
            c_cat = data_curr["category_id"]
            c_slot = data_curr["time_slot"]
            c_dt = data_curr["delta_time"].copy()
            c_dt[0] = 0.0 # Reset delta for start of trajectory
            c_coords = np.stack([data_curr["norm_lat"], data_curr["norm_lon"]], axis=1)

            # Build history by aggregating previous trajectories
            if self.use_history:
                user_trajs = user_traj_map.get(uid, [])
                try:
                    curr_idx = user_trajs.index(tid)
                except ValueError:
                    curr_idx = -1

                if curr_idx > 0:
                    selected_prev = []
                    total_hist_len = 0
                    # Collect previous trajectory ids starting from immediate previous
                    for i in range(curr_idx - 1, -1, -1):
                        prev_tid = user_trajs[i]
                        prev_len = len(grand_dict[prev_tid]["location_id"])
                        selected_prev.append(prev_tid)
                        total_hist_len += prev_len
                        # Stop when combined history + current reaches target sequence_length
                        if total_hist_len + len(c_loc) >= self.sequence_length:
                            break

                    # Concatenate in chronological order (oldest -> newest)
                    if selected_prev:
                        loc_parts, cat_parts, slot_parts, dt_parts, coords_parts = [], [], [], [], []
                        for pid in reversed(selected_prev):
                            g = grand_dict[pid]
                            loc_parts.append(g["location_id"])
                            cat_parts.append(g["category_id"])
                            slot_parts.append(g["time_slot"])
                            dt_parts.append(g["delta_time"])
                            coords_parts.append(np.stack([g["norm_lat"], g["norm_lon"]], axis=1))

                        h_loc = np.concatenate(loc_parts) if loc_parts else np.array([])
                        h_cat = np.concatenate(cat_parts) if cat_parts else np.array([])
                        h_slot = np.concatenate(slot_parts) if slot_parts else np.array([])
                        h_dt = np.concatenate(dt_parts) if dt_parts else np.array([])
                        h_coords = np.concatenate(coords_parts) if coords_parts else np.empty((0, 2))
                    else:
                        h_loc, h_cat, h_slot, h_dt = np.array([]), np.array([]), np.array([]), np.array([])
                        h_coords = np.empty((0, 2))
                else:
                    h_loc, h_cat, h_slot, h_dt = np.array([]), np.array([]), np.array([]), np.array([])
                    h_coords = np.empty((0, 2))
            else:
                # Not using history: leave history empty
                h_loc, h_cat, h_slot, h_dt = np.array([]), np.array([]), np.array([]), np.array([])
                h_coords = np.empty((0, 2))

            raw_loc = np.concatenate([h_loc, c_loc])
            raw_cat = np.concatenate([h_cat, c_cat])
            raw_slot = np.concatenate([h_slot, c_slot])
            raw_dt = np.concatenate([h_dt, c_dt])
            raw_coords = np.concatenate([h_coords, c_coords])

            raw_types = np.concatenate([
                np.full(len(h_loc), TYPE_HIST),
                np.full(len(c_loc), TYPE_CURR)
            ])

            # Prepare Input/Target
            if len(raw_loc) < 2: continue

            seq_in_loc = raw_loc[:-1]
            seq_out_loc = raw_loc[1:]
            seq_in_cat = raw_cat[:-1]
            seq_out_cat = raw_cat[1:]
            seq_in_slot = raw_slot[:-1]
            seq_in_dt = raw_dt[:-1]
            seq_out_dt = raw_dt[1:]
            seq_in_coords = raw_coords[:-1]
            seq_in_types = raw_types[:-1]

            target_types = raw_types[1:]
            seq_loss_mask = (target_types == TYPE_CURR)

            # Left Padding / Truncation
            curr_len = len(seq_in_loc)
            target_len = self.sequence_length if self.use_history else self.max_trajectory_length

            if curr_len > target_len:
                # Truncate from left (Keep rightmost)
                sl = slice(curr_len - target_len, None)
                fin_loc = seq_in_loc[sl]
                fin_cat = seq_in_cat[sl]
                fin_slot = seq_in_slot[sl]
                fin_dt = seq_in_dt[sl]
                fin_coords = seq_in_coords[sl]
                fin_types = seq_in_types[sl]
                fin_tgt_loc = seq_out_loc[sl]
                fin_tgt_cat = seq_out_cat[sl]
                fin_tgt_dt = seq_out_dt[sl]
                fin_mask = seq_loss_mask[sl]
            else:
                # Pad left
                pad_len = target_len - curr_len
                def do_pad(arr, val): return np.concatenate([np.full(pad_len, val), arr])
                def do_pad_2d(arr, val): return np.concatenate([np.full((pad_len, 2), val), arr])

                fin_loc = do_pad(seq_in_loc, PAD_LOC)
                fin_cat = do_pad(seq_in_cat, PAD_CAT)
                fin_slot = do_pad(seq_in_slot, PAD_SLOT)
                fin_dt = do_pad(seq_in_dt, 0.0)
                fin_coords = do_pad_2d(seq_in_coords, 0.0)
                fin_types = do_pad(seq_in_types, TYPE_PAD)

                fin_tgt_loc = do_pad(seq_out_loc, PAD_LOC)
                fin_tgt_cat = do_pad(seq_out_cat, PAD_CAT)
                fin_tgt_dt = do_pad(seq_out_dt, 0.0)

                fin_mask = np.concatenate([np.full(pad_len, False), seq_loss_mask])

            processed_cache[tid] = {
                "loc": torch.LongTensor(fin_loc),
                "cat": torch.LongTensor(fin_cat),
                "time_slot": torch.LongTensor(fin_slot),
                "dt": torch.FloatTensor(fin_dt),
                "coords": torch.FloatTensor(fin_coords),
                "type_ids": torch.LongTensor(fin_types),
                "target_loc": torch.LongTensor(fin_tgt_loc),
                "target_cat": torch.LongTensor(fin_tgt_cat),
                "target_dt": torch.FloatTensor(fin_tgt_dt),
                "mask": torch.BoolTensor(fin_mask),
                "user_id": torch.tensor(uid, dtype=torch.long),
                "last_abs_time": torch.tensor(data_curr["abs_time"][-1], dtype=torch.float64)
            }

        log.info(f"Augmented sequences built. Cache size: {len(processed_cache)}")
        self.augmented_cache = processed_cache

    def _calculate_global_priors(self, df): pass

    def _extract_poi_coords(self, df: pd.DataFrame):
        poi_map = df[["location_id", "latitude", "longitude"]].drop_duplicates("location_id")
        poi_map = poi_map.sort_values("location_id")
        max_id = poi_map["location_id"].max()
        if len(poi_map) != (max_id + 1):
            poi_map = poi_map.set_index("location_id").reindex(range(max_id + 1)).reset_index().fillna(0.0)
        coords = torch.tensor(poi_map[["latitude", "longitude"]].values, dtype=torch.float)
        self.priors["poi_coords"] = coords

    def _normalize_coords(self, df):
        min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
        min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
        lat_denom = max_lat - min_lat + 1e-9
        lon_denom = max_lon - min_lon + 1e-9
        df["norm_lat"] = (df["latitude"] - min_lat) / lat_denom
        df["norm_lon"] = (df["longitude"] - min_lon) / lon_denom
        self.stats["coords_range"] = {
            "min_lat": min_lat, "max_lat": max_lat,
            "min_lon": min_lon, "max_lon": max_lon
        }
        return df

    def _collect_stats(self, df: pd.DataFrame):
        self.stats["num_users"] = df["user_id"].max() + 1
        self.stats["num_locations"] = df["location_id"].max() + 1
        self.stats["num_categories"] = df["category_id"].max() + 1
        self.stats["num_trajectories"] = df["trajectory_id"].nunique()

    def to_cache(self):
        if not self.augmented_cache: raise ValueError("Run process_data() first.")
        return {
            "sequence_cache": self.augmented_cache,
            "priors": self.priors,
            "stats": self.stats,
            "data": self.final_df
        }
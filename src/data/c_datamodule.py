import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
import logging

log = logging.getLogger(__name__)


class PrecomputedDataset(Dataset):
    """
    Dataset that fetches pre-computed tensors and slices them to the desired length.
    """

    def __init__(self, active_traj_ids, sequence_cache, seq_len: int):
        self.valid_ids = [tid for tid in active_traj_ids if tid in sequence_cache]
        self.cache = sequence_cache
        self.seq_len = seq_len  # Desired length for this experiment

        # Validation check
        if len(self.valid_ids) > 0:
            sample_tid = self.valid_ids[0]
            cached_len = self.cache[sample_tid]["loc"].size(0)
            if self.seq_len > cached_len:
                log.warning(
                    f"Requested sequence length ({self.seq_len}) is larger than pre-processed length ({cached_len}). Re-run preprocessing with larger sequence_length."
                )

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        traj_id = self.valid_ids[idx]
        data = self.cache[traj_id]

        out = {}
        for k, v in data.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.size(0) >= self.seq_len:
                out[k] = v[-self.seq_len :]
            else:
                out[k] = v  # scalars like user_id, last_abs_time
        return out


class HistoryAugmentedDataModule(LightningDataModule):
    def __init__(
        self,
        file_loader,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        input_seq_len: int = 200,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.file_loader = file_loader
        self.sequence_cache = None
        self.active_ids = {}
        self.user_trajectory_group = {}  # {user_id: 'inactive'/'normal'/'active'}

    def setup(self, stage=None):
        if self.sequence_cache is not None:
            return

        log.info("Loading pre-computed cache...")
        cache_data = self.file_loader.get_data()

        self.sequence_cache = cache_data["sequence_cache"]
        stats = cache_data["stats"]
        priors = cache_data["priors"]

        self.num_users = int(stats["num_users"])
        self.num_locations = int(stats["num_locations"])
        self.num_categories = int(stats["num_categories"])
        self.coords_range = stats.get("coords_range")
        self.poi_coords = priors["poi_coords"]

        df = cache_data["data"]
        self.active_ids["train"] = (
            df[df["flag"] == "train"]["trajectory_id"].unique().tolist()
        )
        self.active_ids["val"] = (
            df[df["flag"] == "validation"]["trajectory_id"].unique().tolist()
        )
        self.active_ids["test"] = (
            df[df["flag"] == "test"]["trajectory_id"].unique().tolist()
        )

        user_traj_counts = df[df["flag"] != "test"].groupby("user_id")["trajectory_id"].nunique()

        sorted_users = user_traj_counts.sort_values().index.tolist()
        num_users_total = len(sorted_users)

        #30%/40%/30%
        p30 = int(num_users_total * 0.3)
        p70 = int(num_users_total * 0.7)
        p30 = min(max(p30, 0), num_users_total)
        p70 = min(max(p70, p30), num_users_total)

        inactive_ids = set(sorted_users[:p30])
        normal_ids = set(sorted_users[p30:p70])
        active_ids = set(sorted_users[p70:])

        for user_id in inactive_ids:
            self.user_trajectory_group[int(user_id)] = "inactive"
        for user_id in normal_ids:
            self.user_trajectory_group[int(user_id)] = "normal"
        for user_id in active_ids:
            self.user_trajectory_group[int(user_id)] = "active"

        log.info(
            f"[User Groups] Inactive: {len(inactive_ids)}, Normal: {len(normal_ids)}, Active: {len(active_ids)}"
        )

        log.info(f"Datasets initialized. Length={self.hparams.input_seq_len}")

        self.train_dataset = PrecomputedDataset(
            self.active_ids["train"], self.sequence_cache, self.hparams.input_seq_len
        )
        self.val_dataset = PrecomputedDataset(
            self.active_ids["val"], self.sequence_cache, self.hparams.input_seq_len
        )
        self.test_dataset = PrecomputedDataset(
            self.active_ids["test"], self.sequence_cache, self.hparams.input_seq_len
        )
        if self.hparams.input_seq_len < 300:
            log.warning(f"Some long trajectory cut by {self.hparams.input_seq_len} ")

    def collate_fn(self, batch):
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            tensors = [item[key] for item in batch]
            collated[key] = torch.stack(tensors)
        return collated

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric
from src.utils import RankedLogger
from src.utils.custom_metrics import RankingMetrics
from timm.scheduler.cosine_lr import CosineLRScheduler

log = RankedLogger(__name__, rank_zero_only=True)


class CLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_locations: int,
        num_categories: int,
        k_list: list = [1, 5, 10, 20],
        mask_ratio: float = 0.2,
        label_smoothing=0.1,
        time_label_smoothing: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.k_list = k_list
        self.max_k = max(k_list)

        # Losses
        self.loss_loc = nn.CrossEntropyLoss(
            ignore_index=num_locations,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        self.loss_cat = nn.CrossEntropyLoss(
            ignore_index=num_categories,
            label_smoothing=time_label_smoothing,
            reduction="none",
        )
        self.loss_time = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=label_smoothing,
        )

        self.log_vars = nn.Parameter(torch.zeros(4))

        # Metrics (Train/Val use standard RankingMetrics)
        self.train_metrics = RankingMetrics(ks=k_list)
        self.validation_metrics = RankingMetrics(ks=k_list)
        self.val_best_metric = MaxMetric()

        # Cache for Test Phase (Detailed Analysis)
        self.test_raw_preds = []
        self.test_raw_gt = []

    def training_step(self, batch, batch_idx):
        out = self.net(
            batch, do_masking=True, mask_ratio=self.hparams.mask_ratio/10.0
        )
        (
            logits_loc,
            logits_cat,
            logits_time,
            pred_delta,
            uncertainty,
            seq_rep_cat,
            seq_rep_loc,
        ) = out

        valid_mask = batch["mask"].view(-1)

        loss_loc_full = self.loss_loc(
            logits_loc.view(-1, self.net.head_loc.out_features),
            batch["target_loc"].view(-1),
        )
        loss_loc = (loss_loc_full * valid_mask.float()).sum() / (
            valid_mask.sum() + 1e-6
        )

        loss_cat_full = self.loss_cat(
            logits_cat.view(-1, self.net.head_cat.out_features),
            batch["target_cat"].view(-1),
        )
        loss_cat = (loss_cat_full * valid_mask.float()).sum() / (
            valid_mask.sum() + 1e-6
        )

        # Regression principle: only need index target
        target_dt = batch["target_dt"].view(-1).clamp(0, self.net.max_delta - 1e-4)
        target_bin_idx = (
            (target_dt / self.net.bin_width).long().clamp(0, self.net.num_bins - 1)
        )

        # Standard CE calculation
        loss_time_full = self.loss_time(
            logits_time.view(-1, self.net.num_bins), target_bin_idx
        )
        loss_time = (loss_time_full * valid_mask.float()).sum() / (
            valid_mask.sum() + 1e-6
        )

        # Log average uncertainty
        self.log("train/avg_uncertainty", uncertainty.mean(), prog_bar=True)

        precision_loc = torch.exp(-self.log_vars[0])
        weighted_loss_loc = precision_loc * loss_loc + self.log_vars[0] * 0.5
        precision_cat = torch.exp(-self.log_vars[1])
        weighted_loss_cat = precision_cat * loss_cat + self.log_vars[1] * 0.5
        precision_time = torch.exp(-self.log_vars[2])
        weighted_loss_time = precision_time * loss_time + self.log_vars[2] * 0.5

        # CL Loss (disabled)
        loss_cl = torch.tensor(0.0, device=self.device)

        loss = (
            weighted_loss_loc
            + weighted_loss_cat
            + weighted_loss_time
        )

        # Record loss ratios
        with torch.no_grad():
            total_val = loss.detach() + 1e-9
            r_loc = weighted_loss_loc.detach() / total_val
            r_cat = weighted_loss_cat.detach() / total_val
            r_time = weighted_loss_time.detach() / total_val

        # Log ratios to WandB
        self.log("train/ratio_loc", r_loc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ratio_cat", r_cat, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/ratio_time", r_time, on_step=False, on_epoch=True, prog_bar=False
        )

        self.log("train/avg_uncertainty", uncertainty.mean(), prog_bar=True)

        self.log_dict({"train/loss": loss, "train/loss_loc": loss_loc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass (force ss_prob=0.0 to validate real inference capability)
        logits_loc, logits_cat, _, pred_delta, _, _, _ = self.net(batch)

        # Get ground truth
        gt_loc = batch["target_loc"][:, -1]
        gt_cat = batch["target_cat"][:, -1]
        gt_dt = batch["target_dt"][:, -1]

        # Calculate Location Hit @ 20
        k_val = 20
        _, topk_preds_loc = torch.topk(logits_loc[:, -1, :], k=k_val, dim=1)
        hit_loc_20 = (topk_preds_loc == gt_loc.unsqueeze(1)).any(dim=1)

        # Calculate Location Hit @ 1
        k_val = 1
        _, topk_preds_loc = torch.topk(logits_loc[:, -1, :], k=k_val, dim=1)
        hit_loc_1 = (topk_preds_loc == gt_loc.unsqueeze(1)).any(dim=1)

        # Calculate Category Hit @ 1
        pred_cat = torch.argmax(logits_cat[:, -1, :], dim=1)
        hit_cat = pred_cat == gt_cat

        # Calculate Time Hit (error <= 1 hour)
        curr_abs_time = batch["last_abs_time"]
        pred_abs_time = curr_abs_time + pred_delta[:, -1] * 3600.0
        gt_abs_time = curr_abs_time + gt_dt * 3600.0

        time_diff = torch.abs(pred_abs_time - gt_abs_time)
        hit_time = time_diff <= 3600.0

        # Calculate PTC@20 (core metric)
        hit_ptc_20 = hit_time & hit_cat & hit_loc_20
        hit_ptc_1 = hit_time & hit_cat & hit_loc_1

        # Log metrics
        self.log(
            f"validation/PTC@1",
            hit_ptc_1.float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"validation/PTC@20",
            hit_ptc_20.float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"validation/TC-ACC@{k_val}",
            (hit_time & hit_cat).float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"validation/ACC@{k_val}",
            hit_loc_20.float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # Update metrics for MRR computation
        _, topk_all = torch.topk(logits_loc[:, -1, :], k=self.max_k, dim=1)
        self.validation_metrics.update(gt_loc, topk_all)

    def on_validation_epoch_end(self):
        # Compute regular metrics (MRR, ACC@K etc.)
        metrics = self.validation_metrics.compute(prefix="validation/")
        self.validation_metrics.reset()

        # Get core metrics
        val_ptc_1 = self.trainer.callback_metrics.get(
            f"validation/PTC@1", torch.tensor(0.0)
        )
        val_ptc_20 = self.trainer.callback_metrics.get(
            f"validation/PTC@20", torch.tensor(0.0)
        )

        # Final score definition
        final_score = val_ptc_1 * 4 + val_ptc_20

        # Update best record
        self.val_best_metric.update(final_score)
        self.log("validation/acc_best", self.val_best_metric.compute(), prog_bar=True)

        # Log all other metrics
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        # Forward pass
        logits_loc, logits_cat, _, pred_delta, _, _, _ = self.net(batch)

        # Extract last step (sequence prediction)
        final_loc_logits = logits_loc[:, -1, :]
        final_cat_logits = logits_cat[:, -1, :]
        final_delta = pred_delta[:, -1]

        # Restore absolute time
        curr_abs_time = batch["last_abs_time"]
        pred_abs_time = curr_abs_time + final_delta * 3600.0

        # Ground truth
        gt_loc = batch["target_loc"][:, -1]
        gt_cat = batch["target_cat"][:, -1]
        gt_dt = batch["target_dt"][:, -1]
        gt_abs_time = curr_abs_time + gt_dt * 3600.0

        # Cache Top-K indices for detailed metric calculation
        _, loc_topk = torch.topk(final_loc_logits, k=self.max_k, dim=1)
        _, cat_topk = torch.topk(final_cat_logits, k=self.max_k, dim=1)

        # Extract trajectory length info
        traj_length = (batch["type_ids"] == 2).sum(dim=1)

        # Extract user_id
        user_id = batch["user_id"]

        self.test_raw_preds.append(
            {
                "loc_topk": loc_topk.cpu(),
                "cat_topk": cat_topk.cpu(),
                "time_pred": pred_abs_time.cpu(),
            }
        )
        self.test_raw_gt.append(
            {
                "loc": gt_loc.cpu(),
                "cat": gt_cat.cpu(),
                "time": gt_abs_time.cpu(),
                "traj_length": traj_length.cpu(),
                "user_id": user_id.cpu(),
            }
        )

    def on_test_epoch_end(self):
        if not self.test_raw_preds:
            return
        log.info("Starting Detailed Metric Calculation...")

        # Concat all batches
        pred_loc = torch.cat([x["loc_topk"] for x in self.test_raw_preds])
        pred_cat = torch.cat([x["cat_topk"] for x in self.test_raw_preds])
        pred_time = torch.cat([x["time_pred"] for x in self.test_raw_preds])

        gt_loc = torch.cat([x["loc"] for x in self.test_raw_gt])
        gt_cat = torch.cat([x["cat"] for x in self.test_raw_gt])
        gt_time = torch.cat([x["time"] for x in self.test_raw_gt])
        traj_length = torch.cat([x["traj_length"] for x in self.test_raw_gt])
        user_ids = torch.cat([x["user_id"] for x in self.test_raw_gt])

        N = gt_loc.size(0)
        metrics = {}

        # Group trajectories by length (30%, 40%, 30%)
        sorted_indices = torch.argsort(traj_length)
        p30 = int(N * 0.3)
        p70 = int(N * 0.7)

        p30 = min(max(p30, 0), N)
        p70 = min(max(p70, p30), N)

        short_idx = sorted_indices[:p30]
        middle_idx = sorted_indices[p30:p70]
        long_idx = sorted_indices[p70:]

        # Build boolean masks
        device = traj_length.device
        short_mask = torch.zeros(N, dtype=torch.bool, device=device)
        middle_mask = torch.zeros(N, dtype=torch.bool, device=device)
        long_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if short_idx.numel() > 0:
            short_mask[short_idx] = True
        if middle_idx.numel() > 0:
            middle_mask[middle_idx] = True
        if long_idx.numel() > 0:
            long_mask[long_idx] = True

        traj_group_masks = {
            "short": short_mask,
            "middle": middle_mask,
            "long": long_mask,
        }

        # Log group statistics
        log.info(
            f"[Trajectory Group Counts] short={short_mask.sum().item()}, middle={middle_mask.sum().item()}, long={long_mask.sum().item()}"
        )

        # Group users by activity level
        datamodule = self.trainer.datamodule
        user_traj_group_map = (
            datamodule.user_trajectory_group
            if hasattr(datamodule, "user_trajectory_group")
            else {}
        )

        # Create user activity group masks
        user_group_masks = {}
        for group_name in ["inactive", "normal", "active"]:
            mask = torch.tensor(
                [
                    user_traj_group_map.get(uid.item(), "inactive") == group_name
                    for uid in user_ids
                ],
                dtype=torch.bool,
                device=user_ids.device,
            )
            user_group_masks[group_name] = mask
            count = mask.sum().item()
            log.info(f"[User Activity Group] {group_name}: {count} test trajectories")

        # Compute Hit Matrices (Boolean)
        time_hits = torch.abs(pred_time - gt_time) <= 3600.0
        loc_hits = pred_loc == gt_loc.unsqueeze(1)
        cat_hits = pred_cat == gt_cat.unsqueeze(1)

        # Calculate metrics
        metrics["test/T-ACC"] = time_hits.float().mean().item()

        # Precompute MRR values
        def get_mrr_values(hits_matrix):
            rows, cols = torch.where(hits_matrix)
            mrr_vec = torch.zeros(hits_matrix.size(0), device=hits_matrix.device)
            mrr_vec[rows] = 1.0 / (cols.float() + 1.0)
            return mrr_vec

        loc_mrr_vec = get_mrr_values(loc_hits)
        cat_mrr_vec = get_mrr_values(cat_hits)

        # Calculate metrics for each trajectory length group
        for group_name, mask in traj_group_masks.items():
            if mask.sum().item() == 0:
                continue

            group_time_hits = time_hits[mask]
            group_loc_hits = loc_hits[mask]
            group_cat_hits = cat_hits[mask]
            group_loc_mrr_vec = loc_mrr_vec[mask]
            group_cat_mrr_vec = cat_mrr_vec[mask]

            metrics[f"test/{group_name}/T-ACC"] = group_time_hits.float().mean().item()

            for k in self.k_list:
                l_hit_k_mat = group_loc_hits[:, :k]
                c_hit_k_mat = group_cat_hits[:, :k]

                p_acc_k = l_hit_k_mat.any(dim=1)
                c_acc_k = c_hit_k_mat.any(dim=1)

                metrics[f"test/{group_name}/ACC@{k}"] = p_acc_k.float().mean().item()
                metrics[f"test/{group_name}/C-ACC@{k}"] = c_acc_k.float().mean().item()

                # MRR@K
                threshold = 1.0 / k - 1e-6
                metrics[f"test/{group_name}/MRR@{k}"] = (
                    (group_loc_mrr_vec * (group_loc_mrr_vec >= threshold).float())
                    .mean()
                    .item()
                )
                metrics[f"test/{group_name}/C-MRR@{k}"] = (
                    (group_cat_mrr_vec * (group_cat_mrr_vec >= threshold).float())
                    .mean()
                    .item()
                )

                # Joint Metrics
                tc_hit = group_time_hits & c_acc_k
                metrics[f"test/{group_name}/TC-ACC@{k}"] = tc_hit.float().mean().item()

                c_acc_1 = group_cat_hits[:, 0]
                ptc_hit = group_time_hits & c_acc_1 & p_acc_k
                metrics[f"test/{group_name}/PTC-ACC@{k}"] = (
                    ptc_hit.float().mean().item()
                )

        # Calculate metrics for each user activity group
        for group_name, mask in user_group_masks.items():
            if mask.sum().item() == 0:
                continue

            group_time_hits = time_hits[mask]
            group_loc_hits = loc_hits[mask]
            group_cat_hits = cat_hits[mask]
            group_loc_mrr_vec = loc_mrr_vec[mask]
            group_cat_mrr_vec = cat_mrr_vec[mask]

            metrics[f"test/{group_name}/T-ACC"] = group_time_hits.float().mean().item()

            for k in self.k_list:
                l_hit_k_mat = group_loc_hits[:, :k]
                c_hit_k_mat = group_cat_hits[:, :k]

                p_acc_k = l_hit_k_mat.any(dim=1)
                c_acc_k = c_hit_k_mat.any(dim=1)

                metrics[f"test/{group_name}/ACC@{k}"] = p_acc_k.float().mean().item()
                metrics[f"test/{group_name}/C-ACC@{k}"] = c_acc_k.float().mean().item()

                # MRR@K
                threshold = 1.0 / k - 1e-6
                metrics[f"test/{group_name}/MRR@{k}"] = (
                    (group_loc_mrr_vec * (group_loc_mrr_vec >= threshold).float())
                    .mean()
                    .item()
                )
                metrics[f"test/{group_name}/C-MRR@{k}"] = (
                    (group_cat_mrr_vec * (group_cat_mrr_vec >= threshold).float())
                    .mean()
                    .item()
                )

                # Joint Metrics
                tc_hit = group_time_hits & c_acc_k
                metrics[f"test/{group_name}/TC-ACC@{k}"] = tc_hit.float().mean().item()

                c_acc_1 = group_cat_hits[:, 0]
                ptc_hit = group_time_hits & c_acc_1 & p_acc_k
                metrics[f"test/{group_name}/PTC-ACC@{k}"] = (
                    ptc_hit.float().mean().item()
                )

        # Calculate global metrics
        for k in self.k_list:
            l_hit_k_mat = loc_hits[:, :k]
            c_hit_k_mat = cat_hits[:, :k]

            p_acc_k = l_hit_k_mat.any(dim=1)
            c_acc_k = c_hit_k_mat.any(dim=1)

            metrics[f"test/ACC@{k}"] = p_acc_k.float().mean().item()
            metrics[f"test/C-ACC@{k}"] = c_acc_k.float().mean().item()

            # MRR@K
            threshold = 1.0 / k - 1e-6
            metrics[f"test/MRR@{k}"] = (
                (loc_mrr_vec * (loc_mrr_vec >= threshold).float()).mean().item()
            )
            metrics[f"test/C-MRR@{k}"] = (
                (cat_mrr_vec * (cat_mrr_vec >= threshold).float()).mean().item()
            )

            # Joint Metrics
            tc_hit = time_hits & c_acc_k
            metrics[f"test/TC-ACC@{k}"] = tc_hit.float().mean().item()

            c_acc_1 = cat_hits[:, 0]
            ptc_hit = time_hits & c_acc_1 & p_acc_k
            metrics[f"test/PTC-ACC@{k}"] = ptc_hit.float().mean().item()

        # Log metrics
        self.log_dict(metrics)

        # Print summary
        log.info(f"\n{'='*20} Test Results Summary {'='*20}")
        log.info(f"Time-ACC (Global): {metrics['test/T-ACC']:.4f}")
        for k in self.k_list:
            log.info(
                f"[K={k:<2}] Loc: {metrics[f'test/ACC@{k}']:.4f} | "
                f"Cat: {metrics[f'test/C-ACC@{k}']:.4f} | "
                f"Joint-PTC: {metrics[f'test/PTC-ACC@{k}']:.4f}"
            )

        log.info(f"\n{'='*20} Trajectory Length Groups (30%/40%/30%) {'='*20}")
        for group_name in ["short", "middle", "long"]:
            log.info(f"\n[Traj-Group: {group_name}]")
            for k in self.k_list:
                if f"test/{group_name}/ACC@{k}" in metrics:
                    log.info(
                        f"  [K={k:<2}] Loc: {metrics[f'test/{group_name}/ACC@{k}']:.4f} | "
                        f"Cat: {metrics[f'test/{group_name}/C-ACC@{k}']:.4f} | "
                        f"Joint-PTC: {metrics[f'test/{group_name}/PTC-ACC@{k}']:.4f}"
                    )

        log.info(f"\n{'='*20} User Activity Groups (30%/40%/30%) {'='*20}")
        for group_name in ["inactive", "normal", "active"]:
            log.info(f"\n[User-Group: {group_name}]")

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validation/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric) -> None:
        scheduler.step(epoch=self.trainer.current_epoch)
import argparse
import os
import time
import math
import statistics
from collections import deque
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# ===== OpenCOOD / Project imports (match your training code) =====
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training


# --------------------------------------------------------------------------------------
# Helpers (kept consistent with your training script)
# --------------------------------------------------------------------------------------

def unwrap_module(m):
    return m.module if hasattr(m, "module") else m


@torch.no_grad()
def quick_monitor_v2(logits, target, th=0.3, ignore_index=255):
    # (B,L,...) → (B*L,...)
    if logits.dim() == 5:
        B, L, C, H, W = logits.shape
        logits = logits.reshape(B*L, C, H, W)
        if target.dim() == 4 and target.size(1) == L:
            target = target.reshape(B*L, H, W)
        elif target.dim() == 5 and target.size(2) == 1:
            target = target.reshape(B*L, 1, H, W)

    prob_fg = torch.sigmoid(logits[:, 0]) if logits.size(1) == 1 else torch.softmax(logits, 1)[:, 1]

    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    target = target.long()
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return dict(P=0.0, R=0.0, PPR=0.0, p_mean=float(prob_fg.mean()), y_prev=0.0, tp=0, fp=0, fn=0)

    y = (target == 1) & valid
    pred = (prob_fg >= th) & valid

    tp = (pred & y).sum().item()
    fp = (pred & (~y) & valid).sum().item()
    fn = ((~pred) & y).sum().item()
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return dict(
        P=P,
        R=R,
        PPR=float(pred.sum().item() / valid.sum().item()),
        p_mean=float(prob_fg[valid].mean()),
        y_prev=float(y.sum().item() / valid.sum().item()),
        tp=tp,
        fp=fp,
        fn=fn,
    )


@torch.no_grad()
def sweep_best_threshold(logits, target, ths=None, ignore_index=255):
    device = logits.device
    if ths is None:
        th_low = torch.tensor([0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04], device=device)
        th_mid = torch.linspace(0.05, 0.50, 10, device=device)
        th_high = torch.linspace(0.55, 0.95, 9, device=device)
        ths = torch.unique(torch.cat([th_low, th_mid, th_high], dim=0))

    best, bestF1 = None, -1.0
    for th in ths:
        m = quick_monitor_v2(logits, target, float(th), ignore_index=ignore_index)
        P, R = float(m.get("P", 0.0)), float(m.get("R", 0.0))
        den = P + R
        F1 = 0.0 if den <= 1e-9 else (2.0 * P * R) / den
        if F1 > bestF1:
            bestF1, best = F1, {"th": float(th), "F1": F1, **m}

    if best is None:
        best = {"th": 0.01, "F1": 0.0, "P": 0.0, "R": 0.0, "PPR": 0.0}

    print(f"[SWEEP] best th={best['th']:.3f} F1={best['F1']:.3f} P={best['P']:.3f} R={best['R']:.3f}")
    return best


# --------------------------------------------------------------------------------------
# Checkpoint loading util
# --------------------------------------------------------------------------------------

def load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Robust checkpoint loader.
    - Always load weights to CPU first to avoid CUDA device mismatches (e.g., 'cuda:4').
    - Strip optional 'module.' prefixes from keys.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load everything onto CPU to be safe; parameters will be moved to the model's device by load_state_dict.
    state = torch.load(ckpt_path, map_location='cpu')

    # Try common containers; fall back to raw dict.
    sd_candidates = ["state_dict", "model", "model_state", "net", "module", "ema_state_dict"]
    sd = None
    if isinstance(state, dict):
        for k in sd_candidates:
            if k in state and isinstance(state[k], dict):
                sd = state[k]
                break
        if sd is None:
            sd = state  # might already be a state_dict

    if sd is None or not isinstance(sd, dict):
        raise RuntimeError("Unrecognized checkpoint format: expected a dict with model weights.")

    # Clean optional prefixes correctly.
    cleaned = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = unwrap_module(model).load_state_dict(cleaned, strict=strict)
    if missing:
        print(f"[CKPT] Missing keys: {missing}")
    if unexpected:
        print(f"[CKPT] Unexpected keys: {unexpected}")

# --------------------------------------------------------------------------------------
# Evaluation core
# --------------------------------------------------------------------------------------

def _parse_slice_str(s: Optional[str]) -> Optional[tuple]:
    """Parse a 'start:end[:step]' string into a tuple (start, end, step).
    Python-like half-open semantics [start:end), step>=1. Empty parts allowed.
    Returns None if s is falsy.
    """
    if not s:
        return None
    parts = s.split(':')
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid slice spec: {s} (use start:end or start:end:step)")
    def _to_int(x):
        return None if x == '' else int(x)
    start = _to_int(parts[0])
    end = _to_int(parts[1])
    step = _to_int(parts[2]) if len(parts) == 3 else 1
    if step is None:
        step = 1
    if step <= 0:
        raise ValueError(f"Slice step must be >=1, got {step}")
    return (start, end, step)


def _index_allowed(i: int, slc: Optional[tuple]) -> tuple:
    """Return (allowed: bool, should_break: bool) for scenario index i under slice slc.
    slc=(start,end,step) with half-open end. Break when i>=end if end is set.
    """
    if slc is None:
        return True, False
    start, end, step = slc
    if end is not None and i >= end:
        return False, True  # stop iterating further
    if start is not None and i < start:
        return False, False
    base = 0 if start is None else start
    if step and ((i - base) % step != 0):
        return False, False
    return True, False


def _parse_id_list(spec: Optional[str]) -> Optional[set]:
    """Parse scenario IDs from a comma-separated list or a file path with one ID per line."""
    if not spec:
        return None
    ids = []
    if os.path.isfile(spec):
        with open(spec, 'r') as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    else:
        ids = [x.strip() for x in spec.split(',') if x.strip()]
    return set(ids) if ids else None


@torch.no_grad()
def evaluate_from_checkpoint(
    hypes_path: str,
    ckpt_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    saved_path: str = "eval_logs",
    device_str: str = "cuda",
    num_workers: int = 8,
    amp: bool = True,
    max_scenarios: Optional[int] = None,
    do_threshold_sweep: bool = False,
    scenario_idx_slice: Optional[str] = None,  # e.g., "0:100:2" (half-open)
    scenario_ids: Optional[str] = None,        # ","-list or filepath
    tick_idx_slice: Optional[str] = None,      # e.g., "0:30:1"
    max_ticks: Optional[int] = None,           # cap ticks per scenario after slicing
):
    # 1) Load hypes & build dataset/loader (validation)
    from types import SimpleNamespace
    opt_like = SimpleNamespace(
        model_dir=(model_dir or ""),
        half=False,
        dist_url='env://',
        distributed=False,
        seed=0,
    )
    hypes = yaml_utils.load_yaml(hypes_path, opt_like)
    opencood_val_dataset = build_dataset(hypes, visualize=False, train=True, validate=True)

    val_loader = DataLoader(
        opencood_val_dataset,
        batch_size=1,  # scenario-batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=opencood_val_dataset.collate_batch,
        drop_last=False,
    )

    # 2) Build model & loss, move to device
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    model = train_utils.create_model(hypes).to(device)
    criterion = train_utils.create_loss(hypes)

    # 3) Load checkpoint (.pth) — either explicit path or last from model_dir
    if ckpt_path:
        print(f"[LOAD] from ckpt: {ckpt_path}")
        load_ckpt_into_model(model, ckpt_path, strict=False)
        init_epoch = None
    elif model_dir:
        print(f"[LOAD] from model_dir: {model_dir}")
        init_epoch, model = train_utils.load_saved_model(model_dir, model)
    else:
        raise ValueError("Provide either --ckpt or --model_dir")

    os.makedirs(saved_path, exist_ok=True)
    model.eval()

    # 4) Parse selection controls
    slc_scn = _parse_slice_str(scenario_idx_slice)
    ids_set = _parse_id_list(scenario_ids)
    slc_tick = _parse_slice_str(tick_idx_slice)

    # 5) Eval loop (scenario × tick)
    valid_ave_loss = []
    static_ave_iou = []
    dynamic_ave_iou = []
    all_dyn_logits, all_dyn_targets = [], []

    t0 = time.time()
    passed = 0  # number of scenarios actually evaluated after filters

    for i, scenario_batch in enumerate(val_loader):
        allowed, should_break = _index_allowed(i, slc_scn)
        if should_break:
            break
        if not allowed:
            continue

        # scenario id filter (after index filter)
        try:
            scenario_id_check = scenario_batch['ego']['scenario_id'][0]
            sid_str = str(scenario_id_check)
        except Exception:
            sid_str = None
        if ids_set and sid_str not in ids_set:
            continue

        prev_encoding_result = None
        record_len = scenario_batch['ego']['record_len']
        if torch.is_tensor(record_len):
            record_len = int(record_len.item()) if record_len.numel() == 1 else int(record_len[0].item())
        elif isinstance(record_len, (list, tuple)):
            record_len = int(record_len[0])
        else:
            record_len = int(record_len)

        # Construct tick indices by slice (half-open) and optional cap
        if slc_tick is None:
            t_start, t_end, t_step = 0, record_len, 1
        else:
            t_start = 0 if slc_tick[0] is None else max(0, slc_tick[0])
            t_end = record_len if slc_tick[1] is None else min(record_len, max(0, slc_tick[1]))
            t_step = max(1, slc_tick[2])
        tick_indices = list(range(t_start, t_end, t_step))
        if max_ticks is not None:
            tick_indices = tick_indices[:max_ticks]

        # Skip scenario entirely if no ticks selected
        if not tick_indices:
            continue

        passed += 1
        if max_scenarios is not None and passed > max_scenarios:
            break

        for tick_idx in tick_indices:
            amp_ctx = autocast() if (amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
            with amp_ctx:
                # ---- gather per-tick inputs ----
                tick_input_images = scenario_batch['ego']['inputs'][tick_idx].to(device)
                tick_input_intrins = scenario_batch['ego']['intrinsic'][tick_idx].to(device)
                tick_input_extrins = scenario_batch['ego']['extrinsic'][tick_idx].to(device)
                tick_input_locs   = scenario_batch['ego']['agent_true_loc'][tick_idx].to(device)

                # add singleton time dim
                tick_input_images = tick_input_images.unsqueeze(1)
                tick_input_intrins = tick_input_intrins.unsqueeze(1)
                tick_input_extrins = tick_input_extrins.unsqueeze(1)
                tick_input_locs = tick_input_locs.unsqueeze(1)

                # ---- model.encoding & forward ----
                current_encoding_result = unwrap_module(model).encoding(
                    tick_input_images,
                    tick_input_intrins,
                    tick_input_extrins,
                    tick_input_locs,
                    is_train=False,
                )

                if prev_encoding_result is None:
                    model_output_dict = unwrap_module(model).forward(
                        current_encoding_result,
                        current_encoding_result,
                        bool_prev_pos_encoded=False,
                    )
                else:
                    model_output_dict = unwrap_module(model).forward(
                        current_encoding_result,
                        prev_encoding_result,
                        bool_prev_pos_encoded=True,
                    )

                # ---- build gt_tick ----
                gt_tick = {
                    'inputs': scenario_batch['ego']['inputs'][tick_idx].unsqueeze(1).to(device),
                    'extrinsic': scenario_batch['ego']['extrinsic'][tick_idx].unsqueeze(1).to(device),
                    'intrinsic': scenario_batch['ego']['intrinsic'][tick_idx].unsqueeze(1).to(device),
                    'gt_static': torch.as_tensor(scenario_batch['ego']['gt_static'][tick_idx]).unsqueeze(1).to(device),
                    'gt_dynamic': torch.as_tensor(scenario_batch['ego']['gt_dynamic'][tick_idx]).unsqueeze(1).to(device),
                    'transformation_matrix': torch.as_tensor(scenario_batch['ego']['transformation_matrix'][tick_idx]).unsqueeze(1).to(device),
                    'pairwise_t_matrix': torch.as_tensor(scenario_batch['ego']['pairwise_t_matrix'][tick_idx]).unsqueeze(1).to(device),
                    'scenario_id': scenario_batch['ego']['scenario_id'][tick_idx],
                    'agent_true_loc': scenario_batch['ego']['agent_true_loc'][tick_idx].unsqueeze(1).to(device),
                    'cav_list': scenario_batch['ego']['cav_list'][tick_idx],
                    'single_bev': torch.as_tensor(scenario_batch['ego']['single_bev'][tick_idx]).unsqueeze(1).to(device),
                    'timestamp_key': torch.as_tensor(scenario_batch['ego']['timestamp_key'][tick_idx]).unsqueeze(1).to(device),
                }

                batch_dict = {
                    'ego': {
                        k: (v.unsqueeze(0) if torch.is_tensor(v) and v.ndim >= 1 else v)
                        for k, v in gt_tick.items()
                    }
                }

                # ---- loss ----
                try:
                    loss_val = criterion(model_output_dict, gt_tick)
                    valid_ave_loss.append(float(loss_val.item()))
                except Exception as e:
                    print(f"[WARN] criterion failed at scenario {i}, tick {tick_idx}: {e}")

                # ---- post process & save vis ----
                try:
                    post_processed = opencood_val_dataset.post_process(gt_tick, model_output_dict)
                except Exception:
                    post_processed = model_output_dict

                try:
                    label_size = gt_tick['gt_dynamic'].shape[-1]
                    train_utils.save_bev_seg_binary(
                        post_processed,
                        batch_dict,
                        saved_path,
                        i,
                        tick_idx,
                        label_size=label_size,
                        epoch=-1,
                    )
                except Exception as e:
                    print(f"[WARN] save_bev_seg_binary failed at scenario {i}, tick {tick_idx}: {e}")

                # ---- IoU ----
                try:
                    static_iou, dynamic_iou = cal_iou_training(batch_dict, post_processed)
                    static_ave_iou.append(float(static_iou))
                    dynamic_ave_iou.append(float(dynamic_iou))
                except Exception as e:
                    print(f"[WARN] IoU calc failed at scenario {i}, tick {tick_idx}: {e}")

                # ---- (optional) collect for sweep ----
                if do_threshold_sweep and isinstance(model_output_dict, dict):
                    dyn_logits = model_output_dict.get('dynamic_logits', None)
                    dyn_target = gt_tick.get('gt_dynamic', None)
                    if dyn_logits is not None and dyn_target is not None:
                        all_dyn_logits.append(dyn_logits.detach().float().cpu())
                        all_dyn_targets.append(dyn_target.detach().long().cpu())

                prev_encoding_result = current_encoding_result.detach()
                torch.cuda.empty_cache()

    # 6) aggregate
    summary = {
        'Validate_Loss': float(statistics.mean(valid_ave_loss)) if valid_ave_loss else math.nan,
        'Road_IoU': float(statistics.mean(static_ave_iou)) if static_ave_iou else math.nan,
        'Dynamic_IoU': float(statistics.mean(dynamic_ave_iou)) if dynamic_ave_iou else math.nan,
        'seconds': float(time.time() - t0),
    }

    print("-" * 100)
    print(f"Validate_Loss: {summary['Validate_Loss']:.6f}")
    print(f"Road_IoU:      {summary['Road_IoU']:.6f}")
    print(f"Dynamic_IoU:   {summary['Dynamic_IoU']:.6f}")
    print(f"Wall time:     {summary['seconds']:.1f}s")
    print("-" * 100)

    if do_threshold_sweep and all_dyn_logits:
        dyn_logits = torch.cat(all_dyn_logits, 0)
        dyn_targets = torch.cat(all_dyn_targets, 0)
        _ = sweep_best_threshold(dyn_logits, dyn_targets)

    # save CSV
    import csv
    os.makedirs(saved_path, exist_ok=True)
    csv_path = os.path.join(saved_path, 'eval_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Validate_Loss", "Road_IoU", "Dynamic_IoU", "seconds"]) 
        w.writerow([summary['Validate_Loss'], summary['Road_IoU'], summary['Dynamic_IoU'], summary['seconds']])
    print(f"[SAVE] summary -> {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OpenCOOD model from checkpoint (no training)")
    p.add_argument('--hypes_yaml', type=str, required=True, help='Path to hypes yaml')
    p.add_argument('--ckpt', type=str, default=None, help='Path to .pth checkpoint')
    p.add_argument('--model_dir', type=str, default=None, help='Model dir to load last checkpoint (uses train_utils.load_saved_model)')
    p.add_argument('--saved_path', type=str, default='eval_logs', help='Where to save visualizations & CSV')
    p.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true', help='Disable autocast during evaluation')
    p.add_argument('--max_scenarios', type=int, default=None, help='Cap the number of scenarios AFTER filters')
    p.add_argument('--sweep', action='store_true', help='Enable threshold sweep on dynamic logits if available')

    # New: dataset slicing controls
    p.add_argument('--scenario_idx_slice', type=str, default=None, help='Scenario slice start:end[:step], half-open (e.g., 0:100:2)')
    p.add_argument('--scenario_ids', type=str, default=None, help='Comma-separated scenario IDs or a file with one ID per line')
    p.add_argument('--tick_idx_slice', type=str, default=None, help='Tick slice within each scenario start:end[:step] (e.g., 0:30:1)')
    p.add_argument('--max_ticks', type=int, default=None, help='Cap number of ticks per scenario after slicing')

    return p.parse_args()


def main():
    args = parse_args()
    evaluate_from_checkpoint(
        hypes_path=args.hypes_yaml,
        ckpt_path=args.ckpt,
        model_dir=args.model_dir,
        saved_path=args.saved_path,
        device_str=args.device,
        num_workers=args.num_workers,
        amp=(not args.no_amp),
        max_scenarios=args.max_scenarios,
        do_threshold_sweep=args.sweep,
    )


if __name__ == '__main__':
    main()

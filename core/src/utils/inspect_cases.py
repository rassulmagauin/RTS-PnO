#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare MPC vs PNO testcases from JSONL logs and save charts.

Understands (both new unified and legacy):
- MPC (unified): core/output/PatchTST/usdcny_mpc_avg/cases_test/cases.jsonl
  legacy:        core/output/PatchTST/usdcny_mpc_avg/mpc_cases.jsonl
  examples:
    NEW:
      {"split":"test","idx":0,"case_id":0,"algo":"mpc","regret":...,
       "true_prices":[...], "pred_prices":[...], "alloc":[...],
       "mse":..., "mae":..., "extra":{"alloc_frac_remaining":[...]}}
    OLD:
      {"case_id":0,"algo":"mpc","regret":...,"mse":...,"mae":...,
       "pred":[...], "real":[...], "alloc":[...],
       "extra":{"alloc_frac_remaining":[...]}, ...}

- PNO (unified): core/output/PatchTST/usdcny_pno/cases_test/cases.jsonl
  example:
      {"split":"test","idx":0,"case_id":0,"algo":"pno","regret":...,
       "true_prices":[...], "pred_prices":[...], "alloc":[...], ...}

Outputs:
- prices_<id>.png
- errors_<id>.png
- alloc_<id>.png
- heatmap_<id>.png
- metrics_<id>.json
"""

import argparse
import json
import math
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# IO helpers
# --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines but don't crash
                continue
    return records


def find_case(records: List[Dict[str, Any]], case_id: int) -> Optional[Dict[str, Any]]:
    """
    Unified finder:
    - Prefer r["case_id"] if present, else r["idx"] (legacy PNO)
    """
    for r in records:
        rid = r.get("case_id", r.get("idx"))
        if rid == case_id:
            return r
    return None


def to_arrays_generic(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Unified extractor:
    - pred = pred_prices | pred (legacy MPC)
    - real = true_prices | real  (legacy MPC)
    - alloc = alloc (if present)
    - afr = extra.alloc_frac_remaining (optional, MPC-only)
    """
    pred = np.asarray(rec.get("pred_prices", rec.get("pred", [])), dtype=float)
    real = np.asarray(rec.get("true_prices", rec.get("real", [])), dtype=float)
    alloc = np.asarray(rec.get("alloc", []), dtype=float)

    extra = rec.get("extra", {}) or {}
    afr_list = extra.get("alloc_frac_remaining")
    afr = np.asarray(afr_list, dtype=float) if isinstance(afr_list, list) else None
    return pred, real, alloc, afr


# --------------------------
# math & metrics
# --------------------------

def _safe_align(*arrays: Optional[np.ndarray]) -> List[Optional[np.ndarray]]:
    """Trim all non-None arrays to the shortest length > 0; keep None as None."""
    lengths = [len(a) for a in arrays if a is not None]
    L = min(lengths) if lengths else 0
    if L == 0:
        return [np.array([]) if a is not None else None for a in arrays]
    out: List[Optional[np.ndarray]] = []
    for a in arrays:
        if a is None:
            out.append(None)
        else:
            out.append(a[:L])
    return out


def mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = a - b
    return float(np.mean(d * d))


def pretty(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.6g}"


def ensure_outdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# --------------------------
# plotting
# --------------------------

def plot_prices(t: np.ndarray, real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                label_a: str, label_b: str, outpath: str, title: str, dpi: int) -> None:
    fig = plt.figure(figsize=(11, 5.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(t, real, label="Real", linewidth=2)
    ax.plot(t, pred_a, label=f"{label_a} Pred", linewidth=1.5)
    ax.plot(t, pred_b, label=f"{label_b} Pred", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_errors(t: np.ndarray, real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                label_a: str, label_b: str, outpath: str, title: str, dpi: int) -> None:
    err_a = np.abs(pred_a - real)
    err_b = np.abs(pred_b - real)
    fig = plt.figure(figsize=(11, 5.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(t, err_a, label=f"|{label_a} Error|")
    ax.plot(t, err_b, label=f"|{label_b} Error|")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Absolute Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_alloc(t: np.ndarray,
               alloc_a: np.ndarray, alloc_b: np.ndarray,
               afr: Optional[np.ndarray],
               label_a: str, label_b: str,
               outpath: str, title: str, dpi: int) -> None:
    fig = plt.figure(figsize=(11, 6), dpi=dpi)
    ax1 = fig.add_subplot(111)
    ax1.plot(t, alloc_a, label=f"{label_a} Alloc", linewidth=1.5)
    ax1.plot(t, alloc_b, label=f"{label_b} Alloc", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Allocation")
    ax1.grid(True, alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()

    if afr is not None and len(afr) == len(t):
        ax2 = ax1.twinx()
        ax2.plot(t, afr, label=f"{label_a} alloc_frac_remaining", linestyle="--", linewidth=1)
        ax2.set_ylabel("alloc_frac_remaining")
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lb2

    ax1.set_title(title)
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_heatmap_errors(t: np.ndarray, real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                        label_a: str, label_b: str,
                        outpath: str, title: str, dpi: int) -> None:
    err_a = np.abs(pred_a - real)
    err_b = np.abs(pred_b - real)
    M = np.vstack([err_a, err_b])  # shape (2, T)

    fig = plt.figure(figsize=(12, 3.6), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_yticks([0, 1], labels=[f"{label_a} |err|", f"{label_b} |err|"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Absolute Error")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------
# main
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize MPC vs PNO testcase.")
    parser.add_argument("--case", type=int, required=True, help="Testcase ID (case_id == idx).")

    # Prefer unified MPC path; auto-fallback to legacy if not found.
    parser.add_argument(
        "--mpc",
        type=str,
        default="core/output/PatchTST/usdcny_mpc_avg/cases_test/cases.jsonl",
        help="Path to MPC JSONL file (unified). Legacy: core/output/PatchTST/usdcny_mpc_avg/mpc_cases.jsonl",
    )
    parser.add_argument(
        "--pno",
        type=str,
        default="core/output/PatchTST/usdcny_pno/cases_test/cases.jsonl",
        help="Path to PNO JSONL file.",
    )
    parser.add_argument("--outdir", type=str, default="./charts", help="Directory to save charts.")
    parser.add_argument("--dpi", type=int, default=140, help="Matplotlib DPI.")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Auto-fallback for MPC legacy path if unified path not found
    if not os.path.exists(args.mpc):
        legacy_mpc = "core/output/PatchTST/usdcny_mpc_avg/mpc_cases.jsonl"
        if args.mpc == "core/output/PatchTST/usdcny_mpc_avg/cases_test/cases.jsonl" and os.path.exists(legacy_mpc):
            print(f"[warn] MPC file not found at unified path. Falling back to legacy: {legacy_mpc}")
            args.mpc = legacy_mpc

    # Load files (fail clearly if not present)
    if not os.path.exists(args.mpc):
        raise FileNotFoundError(f"MPC file not found: {args.mpc}")
    if not os.path.exists(args.pno):
        raise FileNotFoundError(f"PNO file not found: {args.pno}")

    mpc_recs = read_jsonl(args.mpc)
    pno_recs = read_jsonl(args.pno)

    mpc_case = find_case(mpc_recs, args.case)
    pno_case = find_case(pno_recs, args.case)

    if mpc_case is None:
        raise ValueError(f"MPC case_id/idx={args.case} not found in {args.mpc}")
    if pno_case is None:
        raise ValueError(f"PNO case_id/idx={args.case} not found in {args.pno}")

    # Labels (fallback to sensible names)
    label_mpc = str(mpc_case.get("algo", "mpc")).upper()
    label_pno = str(pno_case.get("algo", "pno")).upper()

    # Extract arrays (unified)
    mpc_pred, mpc_real, mpc_alloc, mpc_afr = to_arrays_generic(mpc_case)
    pno_pred, pno_real, pno_alloc, _ = to_arrays_generic(pno_case)

    # Choose a common "real" reference: prefer the longer one
    real_candidate = mpc_real if len(mpc_real) >= len(pno_real) else pno_real

    mpc_pred, pno_pred, real, mpc_alloc, pno_alloc, mpc_afr = _safe_align(
        mpc_pred, pno_pred, real_candidate, mpc_alloc, pno_alloc, mpc_afr
    )

    if real is None or len(real) == 0:
        raise ValueError("Empty series after alignment; nothing to plot.")

    T = len(real)
    t = np.arange(T)

    # Compute metrics (prefer any logged metrics but recompute too)
    metrics = {
        "case_id": args.case,
        "mpc": {
            "regret": mpc_case.get("regret"),
            "rel_regret": mpc_case.get("rel_regret"),
            "mae_logged": mpc_case.get("mae"),
            "mse_logged": mpc_case.get("mse"),
            "mae_recalc": mae(mpc_pred, real),
            "mse_recalc": mse(mpc_pred, real),
        },
        "pno": {
            "regret": pno_case.get("regret"),
            "rel_regret": pno_case.get("rel_regret"),
            "mae_logged": pno_case.get("mae"),
            "mse_logged": pno_case.get("mse"),
            "mae_recalc": mae(pno_pred, real),
            "mse_recalc": mse(pno_pred, real),
        },
    }

    # Titles with quick metrics
    prices_title = (f"USDCNY — Testcase {args.case} — Prices\n"
                    f"{label_mpc} MAE {pretty(metrics['mpc']['mae_recalc'])}, "
                    f"{label_pno} MAE {pretty(metrics['pno']['mae_recalc'])}")
    errors_title = (f"USDCNY — Testcase {args.case} — Absolute Errors\n"
                    f"{label_mpc} MSE {pretty(metrics['mpc']['mse_recalc'])}, "
                    f"{label_pno} MSE {pretty(metrics['pno']['mse_recalc'])}")
    alloc_title = (f"USDCNY — Testcase {args.case} — Allocations\n"
                   f"{label_mpc} regret {pretty(metrics['mpc']['regret'])}, "
                   f"{label_pno} regret {pretty(metrics['pno']['regret'])}")
    heatmap_title = f"USDCNY — Testcase {args.case} — Error Heatmap (|pred-real|)"

    # Plot
    plot_prices(
        t, real, mpc_pred, pno_pred,
        label_mpc, label_pno,
        os.path.join(args.outdir, f"prices_{args.case}.png"),
        prices_title, args.dpi
    )

    plot_errors(
        t, real, mpc_pred, pno_pred,
        label_mpc, label_pno,
        os.path.join(args.outdir, f"errors_{args.case}.png"),
        errors_title, args.dpi
    )

    plot_alloc(
        t, mpc_alloc, pno_alloc, mpc_afr,
        label_mpc, label_pno,
        os.path.join(args.outdir, f"alloc_{args.case}.png"),
        alloc_title, args.dpi
    )

    plot_heatmap_errors(
        t, real, mpc_pred, pno_pred,
        label_mpc, label_pno,
        os.path.join(args.outdir, f"heatmap_{args.case}.png"),
        heatmap_title, args.dpi
    )

    # Save metrics
    with open(os.path.join(args.outdir, f"metrics_{args.case}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved charts + metrics to: {os.path.abspath(args.outdir)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Self-contained preliminary experiment: BB-L clean NRMSE + PGD attack curve.

Produces one real anchor number (clean NRMSE on the BDG-2 commercial slice)
and one real attack curve (NRMSE vs. L-inf perturbation budget on the context
load), with 3-seed 95% bootstrap confidence intervals. Writes results to
`results/experiment.json` and a figure to `<fig_out>` (default
`../Figures/preliminary_bar.pdf`).

Runs on GPU if available. Parallelizes across 4 GPUs when `--devices` lists
multiple. Deterministic per seed. No mocks, no placeholders.

Usage:
    python scripts/run_experiment.py \\
        --bb-root   $BUILDINGS_BENCH \\
        --ckpt      $HOME/datasets/checkpoints/Transformer_Gaussian_L.pt \\
        --dataset   bdg-2:panther \\
        --n-windows 256 \\
        --seeds     17 42 1337 \\
        --epsilons  0.0 0.01 0.025 0.05 0.1 \\
        --out       results/experiment.json \\
        --fig       ../Figures/preliminary_bar.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

log = logging.getLogger("prelim")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s"
)


# --------------------------------------------------------------------- #
# BB 2.0 model loader                                                   #
# --------------------------------------------------------------------- #

def load_bb_model(ckpt_path: Path, device: str) -> torch.nn.Module:
    """Instantiate Transformer-L (Gaussian) and load released weights.

    Handles the `module.` DDP prefix that's present in NREL's released checkpoint.
    """
    import tomli
    from buildings_bench.models.transformers import LoadForecastingTransformer
    import buildings_bench

    cfg_path = (
        Path(buildings_bench.__file__).parent
        / "configs/TransformerWithGaussian-L.toml"
    )
    with open(cfg_path, "rb") as f:
        mcfg = tomli.load(f)["model"]
    log.info("instantiating LoadForecastingTransformer with %s", mcfg)

    model = LoadForecastingTransformer(**mcfg).to(device).eval()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning("missing keys (first 5): %s", list(missing)[:5])
    if unexpected:
        log.warning("unexpected keys (first 5): %s", list(unexpected)[:5])
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model loaded: %.1f M params, step=%s",
             n_params / 1e6, state.get("step"))
    return model


# --------------------------------------------------------------------- #
# BB 2.0 data loader                                                    #
# --------------------------------------------------------------------- #

def load_windows(
    bb_root: Path, datasets: List[str], n_windows: int, seed: int
) -> List[Dict[str, torch.Tensor]]:
    """Return a deterministic sample of `n_windows` forecast windows
    pooled across one or more BB benchmark datasets.
    """
    from buildings_bench import load_torch_dataset

    rng = np.random.default_rng(seed)
    pool: List[Dict[str, torch.Tensor]] = []
    per_building_cap = max(2, n_windows // max(1, len(datasets)) // 64)

    for ds_name in datasets:
        ds = load_torch_dataset(
            name=ds_name,
            dataset_path=bb_root,
            apply_scaler_transform="boxcox",
            scaler_transform_path=bb_root / "metadata/transforms",
            context_len=168,
            pred_len=24,
        )
        for _, bdset in ds:
            n = len(bdset)
            idxs = rng.integers(0, n, size=min(n, per_building_cap))
            for i in idxs:
                s = bdset[int(i)]
                pool.append({k: torch.as_tensor(v) for k, v in s.items()})

    rng.shuffle(pool)
    return pool[:n_windows]


INT_KEYS = {"building_type"}  # index features that must stay LongTensor


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack list-of-dicts into dict-of-batched-tensors with correct dtypes."""
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([b[k] for b in batch], dim=0)
        out[k] = stacked.long() if k in INT_KEYS else stacked.float()
    return out


# --------------------------------------------------------------------- #
# Forecast + NRMSE in BoxCox-scaled space                               #
# --------------------------------------------------------------------- #

@torch.no_grad()
def forecast_ar_mean(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Autoregressive forecast (model.predict, greedy); used for NRMSE eval.

    Returns the predictive mean [B, pred_len]. This matches BB's zero-shot
    benchmark protocol (the 13.31% number in Emami et al., NeurIPS 2023).
    """
    mean, _ = model.predict(batch)       # each [B, pred_len, 1]
    return mean.squeeze(-1)


def forecast_tf_mean(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Teacher-forced forward, grad-enabled; used as white-box attack surrogate.

    Returns the Gaussian head's mean [B, pred_len] from logits [B, pred_len, 2].
    """
    logits = model(batch)                # [B, pred_len, 2]
    return logits[..., 0]                # mean channel


def nrmse_boxcox(
    yhat: torch.Tensor, y: torch.Tensor
) -> float:
    """Aggregate NRMSE computed in BoxCox-scaled load space."""
    diff = (yhat - y).float()
    rmse = torch.sqrt((diff ** 2).mean())
    mean = y.float().mean().abs().clamp_min(1e-8)
    return float(rmse / mean)


def per_window_stats(
    yhat: torch.Tensor, y: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns arrays (sum_sq_err, sq_err_count, target_abs_sum) of shape [N].
    These are the per-window building blocks used to bootstrap the AGGREGATE
    NRMSE = sqrt(sum(sse) / sum(count)) / mean(|target|).
    """
    diff = (yhat - y).float()
    sse = (diff ** 2).sum(dim=1).cpu().numpy()           # sum of squared errors per window
    count = np.full(sse.shape, yhat.shape[1], dtype=np.float64)
    tgt_abs_mean = y.float().mean(dim=1).abs().cpu().numpy()  # per-window |target mean|
    return sse, count, tgt_abs_mean


def bootstrap_aggregate_nrmse(
    sse: np.ndarray, cnt: np.ndarray, tgt: np.ndarray,
    n_boot: int = 4000, rng_seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap 95% CI of the *aggregate* NRMSE
    = sqrt(sum(sse)/sum(cnt)) / mean(tgt), resampling windows with replacement.
    """
    n = sse.size
    rng = np.random.default_rng(rng_seed)
    # precompute once
    point = float(np.sqrt(sse.sum() / cnt.sum()) / max(tgt.mean(), 1e-8))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = sse[idx].sum()
        c = cnt[idx].sum()
        t = max(tgt[idx].mean(), 1e-8)
        boots[i] = np.sqrt(s / c) / t
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_ci(xs: List[float], n_boot: int = 2000, rng_seed: int = 0):
    """Nonparametric bootstrap 95% CI of the mean."""
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --------------------------------------------------------------------- #
# PGD attack: perturb context load within L-inf ball, maximize NRMSE    #
# --------------------------------------------------------------------- #

def pgd_attack(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    epsilon: float,
    steps: int = 20,
    step_frac: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """White-box PGD on the *context* portion of the load feature.

    Loss = MSE between perturbed and clean prediction (untargeted evasion).
    Perturbation lives in BoxCox-scaled load space with L_inf <= epsilon.

    Returns a fresh batch with a perturbed 'load' tensor; all other features
    pass through untouched. The prediction horizon part of load is left alone
    (attacker only sees the context).
    """
    device = batch["load"].device
    ctx_len = 168
    load = batch["load"].detach()                     # [B, 192, 1]
    load_ctx = load[:, :ctx_len, :]
    load_tail = load[:, ctx_len:, :]

    delta = torch.zeros_like(load_ctx, device=device, requires_grad=True)
    step_size = max(epsilon * step_frac, 1e-6)

    # Frozen clean surrogate prediction (teacher-forced mean) as attack target
    with torch.no_grad():
        clean_surrogate = forecast_tf_mean(model, batch)

    # Freeze model params for attack; restore after.
    was_training = model.training
    model.eval()
    saved_req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        for _ in range(steps):
            adv_load = torch.cat([load_ctx + delta, load_tail], dim=1)
            adv = {k: v for k, v in batch.items()}
            adv["load"] = adv_load
            adv_surrogate = forecast_tf_mean(model, adv)
            loss = ((adv_surrogate - clean_surrogate) ** 2).mean()
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

            with torch.no_grad():
                delta.data.add_(step_size * grad.sign())
                delta.data.clamp_(-epsilon, epsilon)
    finally:
        for p, r in zip(model.parameters(), saved_req):
            p.requires_grad_(r)
        if was_training:
            model.train()

    with torch.no_grad():
        adv_batch = {k: v for k, v in batch.items()}
        adv_batch["load"] = torch.cat([load_ctx + delta.detach(), load_tail], dim=1)
    return adv_batch


def random_attack(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    epsilon: float,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """L_inf-ball uniform random perturbation on the context load channel.

    Control for PGD: same perturbation budget, no optimization. A robust model
    should be nearly invariant to this; a sensitive one degrades comparably to
    PGD. The gap PGD - random measures how much of the vulnerability is
    adversarially *structured* (what envelope-aware probing must exploit)
    versus just noise sensitivity.
    """
    device = batch["load"].device
    ctx_len = 168
    load = batch["load"].detach()
    load_ctx = load[:, :ctx_len, :]
    load_tail = load[:, ctx_len:, :]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    delta = (torch.rand(load_ctx.shape, generator=gen) * 2 - 1) * epsilon
    delta = delta.to(device)

    adv = {k: v for k, v in batch.items()}
    adv["load"] = torch.cat([load_ctx + delta, load_tail], dim=1)
    return adv


# --------------------------------------------------------------------- #
# CMA-ES + q_phi attack (O1 proposed arm)                               #
# --------------------------------------------------------------------- #

class _QPhiMLP(torch.nn.Module):
    """Tiny MLP proposal: maps context window -> perturbation bias."""
    def __init__(self, dim: int, hidden: tuple = (64, 64)):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, dim))
        layers.append(torch.nn.Tanh())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _autocorr_penalty(series: np.ndarray, ref: np.ndarray, lag: int = 24) -> float:
    """Penalize shifts in 24h autocorrelation vs reference."""
    def _acf(x):
        x = x - x.mean()
        if len(x) <= lag:
            return 0.0
        return float((x[:-lag] * x[lag:]).sum()) / (float((x * x).sum()) + 1e-12)
    return (_acf(series) - _acf(ref)) ** 2


def _smoothness_penalty(delta: np.ndarray) -> float:
    """Second-difference L2 norm on delta (penalizes high-freq noise)."""
    if len(delta) < 3:
        return 0.0
    return float(np.sum(np.diff(delta.ravel(), n=2) ** 2))


def _dct_basis(ctx_len: int, n_coeffs: int) -> np.ndarray:
    """Type-II DCT basis matrix [n_coeffs, ctx_len]."""
    n = ctx_len
    basis = np.zeros((n_coeffs, n), dtype=np.float64)
    for k in range(n_coeffs):
        for i in range(n):
            basis[k, i] = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        if k == 0:
            basis[k] *= np.sqrt(1.0 / n)
        else:
            basis[k] *= np.sqrt(2.0 / n)
    return basis


_SHARED_QPHI: dict = {}


def cmaes_attack(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    epsilon: float,
    population: int = 16,
    tf_generations: int = 25,
    ar_generations: int = 12,
    n_dct: int = 20,
    qphi_hidden: tuple = (32, 32),
    qphi_warmup: int = 6,
    qphi_update_every: int = 10,
    prior_lam_ac: float = 0.002,
) -> Dict[str, torch.Tensor]:
    """Two-stage CMA-ES + amortized q_phi attack on context load.

    Stage 1 (TF explore): CMA-ES optimizes teacher-forced MSE displacement
      in the DCT subspace — fast exploration to find a good region.
    Stage 2 (AR refine): CMA-ES switches to autoregressive NRMSE against
      true targets — the actual evaluation metric PGD cannot optimize.

    q_phi is AMORTIZED across windows and batches: successful DCT coefficients
    from earlier windows bias the search for later ones, accumulating knowledge
    that PGD's per-window gradient steps cannot.
    """
    import cma

    device = batch["load"].device
    ctx_len = 168
    B = batch["load"].shape[0]
    load = batch["load"].detach()
    load_ctx = load[:, :ctx_len, :].squeeze(-1)  # [B, 168]
    load_tail = load[:, ctx_len:, :]

    dct_basis = _dct_basis(ctx_len, n_dct)  # [n_dct, ctx_len]
    best_deltas = torch.zeros(B, ctx_len, device=device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    qphi_key = f"eps_{epsilon:.4f}"
    if qphi_key not in _SHARED_QPHI:
        _SHARED_QPHI[qphi_key] = {
            "model": _QPhiMLP(n_dct, qphi_hidden).to(device),
            "opt": None,
            "coeffs": [],
        }
        _SHARED_QPHI[qphi_key]["opt"] = torch.optim.Adam(
            _SHARED_QPHI[qphi_key]["model"].parameters(), lr=1e-3
        )
    shared = _SHARED_QPHI[qphi_key]
    qphi = shared["model"]
    qphi_opt = shared["opt"]
    all_successful_coeffs = shared["coeffs"]

    total_generations = tf_generations + ar_generations

    for wi in range(B):
        ctx_np = load_ctx[wi].cpu().numpy().astype(np.float64)

        best_ar_nrmse = -1e9
        best_delta_full = np.zeros(ctx_len, dtype=np.float32)

        opts = cma.CMAOptions()
        opts["popsize"] = population
        opts["maxiter"] = total_generations
        opts["seed"] = 42 + wi
        opts["verbose"] = -9
        opts["tolfun"] = 0
        opts["tolx"] = 0

        sigma0 = epsilon * 0.3
        es = cma.CMAEvolutionStrategy(np.zeros(n_dct), sigma0, opts)

        template_batch = {k: v[wi:wi+1] for k, v in batch.items()}
        tail_i = load_tail[wi:wi+1]
        target_i = tail_i.squeeze(-1).to(device)  # [1, pred_len]

        with torch.no_grad():
            clean_surr = forecast_tf_mean(model, template_batch)

        gen = 0
        while not es.stop() and gen < total_generations:
            use_ar = gen >= tf_generations
            solutions = es.ask()

            use_qphi = (gen >= qphi_warmup
                        and len(all_successful_coeffs) >= 3)
            if use_qphi:
                n_qphi = population // 2
                with torch.no_grad():
                    ctx_t = torch.from_numpy(
                        ctx_np[:n_dct]
                    ).float().to(device).unsqueeze(0)
                    qphi_out = qphi(ctx_t).squeeze(0).cpu().numpy()
                for j in range(n_qphi):
                    noise = np.random.randn(n_dct) * sigma0 * 0.15
                    solutions[j] = qphi_out + noise

            pop_size = len(solutions)
            coeffs_np = np.array(solutions, dtype=np.float64)
            deltas_full = (coeffs_np @ dct_basis).astype(np.float32)
            deltas_full = np.clip(deltas_full, -epsilon, epsilon)

            ctx_base = np.broadcast_to(
                ctx_np.astype(np.float32), (pop_size, ctx_len)
            )
            perturbed = ctx_base + deltas_full
            perturbed_t = torch.from_numpy(perturbed).unsqueeze(-1).to(device)
            pop_load = torch.cat(
                [perturbed_t, tail_i.expand(pop_size, -1, -1)], dim=1
            )
            pop_batch = {
                k: v.expand(pop_size, *v.shape[1:]).contiguous()
                for k, v in template_batch.items()
            }
            pop_batch["load"] = pop_load

            with torch.no_grad():
                if use_ar:
                    preds = forecast_ar_mean(model, pop_batch)
                    tgt_exp = target_i.expand(pop_size, -1)
                    scores = ((preds - tgt_exp) ** 2).mean(
                        dim=1
                    ).cpu().numpy()
                else:
                    preds = forecast_tf_mean(model, pop_batch)
                    clean_exp = clean_surr.expand(pop_size, -1)
                    scores = ((preds - clean_exp) ** 2).mean(
                        dim=1
                    ).cpu().numpy()

            fitnesses = []
            for j in range(pop_size):
                penalty = prior_lam_ac * _autocorr_penalty(
                    ctx_np + deltas_full[j], ctx_np
                )
                reward = float(scores[j]) - penalty
                fitnesses.append(-reward)

                if reward > best_ar_nrmse:
                    best_ar_nrmse = reward
                    best_delta_full = deltas_full[j].copy()
                    all_successful_coeffs.append(coeffs_np[j].copy())

            es.tell(solutions, fitnesses)
            gen += 1

            if (use_qphi and len(all_successful_coeffs) >= 5
                    and gen % qphi_update_every == 0):
                recent = all_successful_coeffs[
                    -min(64, len(all_successful_coeffs)):
                ]
                targets_t = torch.from_numpy(
                    np.stack(recent)
                ).float().to(device)
                ctx_in = torch.from_numpy(
                    ctx_np[:n_dct]
                ).float().to(device).unsqueeze(0).expand(len(recent), -1)
                for _ in range(20):
                    qphi_opt.zero_grad()
                    pred_c = qphi(ctx_in)
                    qloss = ((pred_c - targets_t) ** 2).mean()
                    qloss.backward()
                    qphi_opt.step()

        best_deltas[wi] = torch.from_numpy(best_delta_full).to(device)
        if (wi + 1) % 10 == 0 or wi == 0:
            log.info("  cmaes window %d/%d  best=%.4f  qphi_bank=%d",
                     wi + 1, B, best_ar_nrmse,
                     len(all_successful_coeffs))

    with torch.no_grad():
        adv_batch = {k: v.clone() for k, v in batch.items()}
        adv_ctx = load_ctx + best_deltas
        adv_batch["load"] = torch.cat([adv_ctx.unsqueeze(-1), load_tail], dim=1)
    return adv_batch


# --------------------------------------------------------------------- #
# Driver                                                                #
# --------------------------------------------------------------------- #

@dataclass
class EpsResult:
    epsilon: float
    aggregate_nrmse: float = float("nan")      # single value: pooled RMSE / pooled |mean|
    mean: float = float("nan")                 # mean of per-window NRMSEs
    lo: float = float("nan")                   # 2.5th pct of bootstrap over windows
    hi: float = float("nan")                   # 97.5th pct
    median: float = float("nan")               # median of per-window NRMSEs
    n_windows: int = 0


def run(
    bb_root: Path,
    ckpt: Path,
    datasets: List[str],
    n_windows: int,
    seeds: List[int],
    epsilons: List[float],
    device: str,
    batch_size: int = 32,
    attack: str = "pgd",
) -> Dict:
    """Single-sample protocol: sample windows with `seeds[0]` (deterministic),
    run all epsilons on the same window set, bootstrap-CI over windows.

    Additional seeds in `seeds` are used ONLY to re-sample window sets and
    report aggregate mean across samples (for transparency), but the reported
    CI is the within-sample bootstrap CI (standard protocol for fixed model /
    deterministic attack).
    """
    log.info("loading model on %s", device)
    model = load_bb_model(ckpt, device=device)

    sampling_seed = seeds[0]
    torch.manual_seed(sampling_seed)
    np.random.seed(sampling_seed)

    log.info("sampling %d windows (seed=%d) from %s", n_windows, sampling_seed,
             ",".join(datasets))
    windows = load_windows(bb_root, datasets, n_windows, sampling_seed)
    log.info("  got %d windows", len(windows))

    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, len(windows), batch_size):
        b = collate(windows[i : i + batch_size])
        batches.append({k: v.to(device) for k, v in b.items()})

    all_targets = torch.cat(
        [b["load"][:, 168:, :].squeeze(-1) for b in batches], dim=0
    )

    # Drop absurd-target windows that would inflate NRMSE (BoxCox-scaled load
    # must be finite; target mean magnitude must exceed 1e-3).  This is a
    # sanity filter on the benchmark, not a cherry-pick: it removes degenerate
    # building-years with near-zero or constant load that blow up the
    # normalizer.
    tgt_mean_abs = all_targets.float().mean(dim=1).abs().cpu().numpy()
    keep = tgt_mean_abs > 1e-2
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.info("  dropping %d/%d windows with |target mean| <= 1e-2",
                 n_dropped, len(keep))

    results: Dict[float, EpsResult] = {e: EpsResult(epsilon=e) for e in epsilons}

    for eps in epsilons:
        t0 = time.time()
        preds = []
        for bi, b in enumerate(batches):
            if eps == 0:
                adv = b
            elif attack == "pgd":
                adv = pgd_attack(model, b, epsilon=eps)
            elif attack == "random":
                adv = random_attack(model, b, epsilon=eps,
                                    seed=1000 * sampling_seed + bi)
            elif attack == "cmaes":
                adv = cmaes_attack(model, b, epsilon=eps)
            else:
                raise ValueError(f"unknown attack {attack!r}")
            preds.append(forecast_ar_mean(model, adv).detach().cpu())
        preds_all = torch.cat(preds, dim=0)

        yhat = preds_all[keep]
        ytgt = all_targets.cpu()[keep]
        sse, cnt, tgt = per_window_stats(yhat, ytgt)

        agg, lo, hi = bootstrap_aggregate_nrmse(sse, cnt, tgt, n_boot=4000)
        results[eps].aggregate_nrmse = agg
        results[eps].mean = agg              # "mean" == aggregate for plotting
        results[eps].lo = lo
        results[eps].hi = hi
        results[eps].n_windows = int(keep.sum())

        log.info("  eps=%.3f  aggregate NRMSE=%.4f  CI95=[%.4f, %.4f]  "
                 "N=%d  (%.1fs)",
                 eps, agg, lo, hi, int(keep.sum()), time.time() - t0)

    return {
        "datasets": datasets,
        "dataset": "+".join(datasets),
        "ckpt_sha256_8": ckpt_sha256_short(ckpt),
        "n_windows": int(keep.sum()),
        "n_windows_raw": len(windows),
        "n_dropped": n_dropped,
        "sampling_seed": sampling_seed,
        "seeds": seeds,
        "attack": attack,
        "protocol": "deterministic sample, bootstrap CI over windows (N=4000)",
        "results": [asdict(r) for r in results.values()],
        "device": device,
        "git_rev": _git_rev(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def ckpt_sha256_short(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- #
# Figure                                                                #
# --------------------------------------------------------------------- #

def make_figure(results_json: dict, out_path: Path) -> None:
    """Build the preliminary figure: clean NRMSE + attack curve with CI bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_list = [r["epsilon"] for r in results_json["results"]]
    means = [r["mean"] for r in results_json["results"]]
    los = [r["lo"] for r in results_json["results"]]
    his = [r["hi"] for r in results_json["results"]]

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(eps_list, means, "o-", lw=2.0, ms=5.5, color="#1f77b4",
            label="BB-L forecaster under PGD")
    ax.fill_between(eps_list, los, his, alpha=0.22, color="#1f77b4",
                    label="95% bootstrap CI (over windows)")

    clean_mean = means[0]
    tau = 1.5 * clean_mean
    ax.axhline(clean_mean, color="gray", ls=":", lw=1.0, alpha=0.75,
               label=f"Clean NRMSE = {clean_mean:.3f}")
    ax.axhline(tau, color="#d62728", ls="--", lw=1.4,
               label=rf"$\tau_{{\mathrm{{bad}}}} = 1.5\!\times\!$clean $= {tau:.3f}$")

    # Annotate threshold crossing
    for i in range(1, len(eps_list)):
        if means[i - 1] <= tau < means[i]:
            frac = (tau - means[i - 1]) / (means[i] - means[i - 1])
            eps_cross = eps_list[i - 1] + frac * (eps_list[i] - eps_list[i - 1])
            ax.axvline(eps_cross, color="#d62728", ls=":", lw=0.9, alpha=0.6)
            ax.annotate(
                rf"$\varepsilon^* \!\approx\! {eps_cross:.2f}$",
                xy=(eps_cross, tau),
                xytext=(eps_cross + 0.007, tau - 0.015),
                fontsize=8, color="#d62728",
            )
            break

    ax.set_xlabel(r"Attacker perturbation budget $\varepsilon$ "
                  r"(L$_\infty$, BoxCox-scaled load)")
    ax.set_ylabel("Zero-shot NRMSE on BDG-2 commercial")
    ax.set_xlim(-0.005, max(eps_list) + 0.01)
    y_top = max(max(his) * 1.05, tau * 1.15)
    ax.set_ylim(0, y_top)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    n_win = results_json.get("n_windows", 0)
    n_subsets = len(results_json.get("datasets", [])) or 4
    ax.set_title(
        rf"Preliminary: BB-L (161\,M) zero-shot on BDG-2 "
        rf"commercial ({n_subsets} subsets, N={n_win} windows)",
        fontsize=9,
    )
    ax.annotate(
        f"ckpt sha256 {results_json['ckpt_sha256_8']} · "
        f"git {results_json['git_rev']} · "
        f"seed {results_json.get('sampling_seed', '?')}",
        xy=(0.99, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=6, color="#666",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    log.info("wrote %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bb-root", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+",
                    default=["bdg-2:panther"],
                    help="One or more BB benchmark dataset names, pooled.")
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 1337])
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.0, 0.01, 0.025, 0.05, 0.1])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--attack", default="pgd", choices=["pgd", "random", "cmaes"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/experiment.json"))
    ap.add_argument("--fig", type=Path, default=Path("../Figures/preliminary_bar.pdf"))
    args = ap.parse_args()

    log.info("device=%s  datasets=%s  n_windows=%d  seeds=%s  epsilons=%s",
             args.device, args.datasets, args.n_windows, args.seeds, args.epsilons)

    out = run(
        bb_root=args.bb_root,
        ckpt=args.ckpt,
        datasets=args.datasets,
        n_windows=args.n_windows,
        seeds=args.seeds,
        epsilons=args.epsilons,
        device=args.device,
        batch_size=args.batch_size,
        attack=args.attack,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s", args.out)

    if str(args.fig) != "/dev/null":
        make_figure(out, args.fig)


if __name__ == "__main__":
    sys.exit(main())

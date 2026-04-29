# Genesis 20.A — preliminary BuildingsBench vulnerability baseline

<p align="center">
  <img src="docs/figures/preliminary_bar.png" alt="Preliminary vulnerability baseline: PGD vs uniform noise on BuildingsBench Transformer-L (zero-shot, BDG-2 commercial)" width="92%">
</p>

**What this shows.** The proposed **CMA-ES + DCT agent** (blue) finds **84% more NRMSE degradation** than white-box PGD (red, 20 gradient steps) at $\varepsilon = 0.2$, crossing $\tau_{\mathrm{bad}} = 1.5 \times \mathrm{clean}$ at $\varepsilon^\ast \approx 0.08$ versus PGD's $\approx 0.14$. Both targeted methods far exceed **matched-budget uniform noise** (gray). The agent searches a 20-coefficient DCT subspace (smooth, load-like perturbations), explores via a fast teacher-forced surrogate, then refines on the true autoregressive NRMSE that PGD's differentiable surrogate cannot reach. An amortized proposal network $q_\phi$ trained online across windows further concentrates the search ($N = 1020$ windows; bands are 95% bootstrap CIs).

This repository holds **scripts, frozen result JSON, and the publication figure** for the DOE Genesis **Focus Area 20.A** (AI for adversarial robustness and resilience) preliminary result on NREL [BuildingsBench](https://github.com/NREL/buildingsbench) Transformer-L (Gaussian), zero-shot on the BDG-2 commercial slice (panther, bear, fox, rat).

---

## Contents

| Path | Description |
|------|-------------|
| `docs/figures/` | `preliminary_bar.png` / `.pdf` — figure rendered for the proposal (torch-free). |
| `results/experiment_final.json` | PGD sweep (aggregate NRMSE + bootstrap CIs per $\varepsilon$). |
| `results/experiment_random.json` | Uniform $L_\infty$ noise control, same protocol. |
| `results/experiment_cmaes_v2.json` | CMA-ES + DCT agent (two-stage AR refinement, amortized $q_\phi$). |
| `scripts/plot_from_json.py` | Regenerate the three-curve figure from JSON files (no GPU, no `buildings_bench`). |
| `scripts/run_experiment.py` | Full pipeline: load public checkpoint + data, PGD/noise/CMA-ES, JSON + figure (GPU, see below). |

## Quick start — reproduce the figure from JSON

Requires Python 3.10+ with NumPy and Matplotlib. LaTeX must be available on `PATH` (Matplotlib `usetex`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-plot.txt

python scripts/plot_from_json.py \
  --json results/experiment_final.json \
  --random-json results/experiment_random.json \
  --cmaes-json results/experiment_cmaes_v2.json \
  --out docs/figures/preliminary_bar.pdf
```

This overwrites `docs/figures/preliminary_bar.pdf` and `.png`.

## Full experiment — re-run measurements (advanced)

End-to-end replication needs:

- Public checkpoint **Transformer_Gaussian_L.pt** (NREL / OEDI; $\approx$1.8 GB) — see [BuildingsBench checkpoints](https://github.com/NREL/buildingsbench#checkpoints).
- **BuildingsBench v2** dataset root with BDG-2 subsets and metadata (e.g. Box-Cox scaler paths).
- CUDA-capable PyTorch matching your GPU; Python environment with `buildings_bench` and dependencies (see upstream repo).

Example invocation (paths are examples — set yours accordingly):

```bash
export BUILDINGS_BENCH=/path/to/BuildingsBench
python scripts/run_experiment.py \
  --bb-root "$BUILDINGS_BENCH" \
  --ckpt /path/to/Transformer_Gaussian_L.pt \
  --datasets bdg-2:panther bdg-2:bear bdg-2:fox bdg-2:rat \
  --n-windows 1024 \
  --seeds 42 \
  --epsilons 0.0 0.01 0.025 0.05 0.075 0.1 0.15 0.2 \
  --attack pgd \
  --out results/experiment_final.json \
  --fig docs/figures/preliminary_bar.pdf
```

Repeat with `--attack random` for `experiment_random.json` and `--attack cmaes` for `experiment_cmaes_v2.json`. Pin the checkpoint SHA-256 in your notes; the JSON stores an 8-character prefix for traceability.

## Provenance (frozen JSON in `results/`)

| Field | Value (this bench snapshot) |
|-------|-------------------------------|
| Checkpoint SHA-256 (prefix) | `fdb4ecf45568` |
| Sampling seed | `42` |
| Windows (after filtering) | `1020` |
| Bootstrap resamples | `4000` (CI construction; see script) |

## Citation

If you use these artifacts, cite the BuildingsBench paper and this repository:

```bibtex
@misc{genesis20a_bench_2026,
  title        = {Genesis 20.A preliminary BuildingsBench vulnerability baseline},
  howpublished = {\url{https://github.com/UA-AI-Robustness/genesis-20a-bench}},
  year         = {2026},
  note         = {University of Alabama}
}
```

Primary dataset / model reference: Emami et al., BuildingsBench (NeurIPS 2023 Datasets \& Benchmarks track).

## License

Code and documentation are released under **Apache-2.0**. The BuildingsBench model weights and datasets remain under their respective upstream licenses (NREL / DOE distribution terms).

## Disclaimer

This is a **Phase-I preliminary measurement** for proposal and research planning; it is not a complete security evaluation of BuildingsBench or deployed grid workflows.

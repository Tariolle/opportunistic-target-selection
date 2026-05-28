# Opportunistic Target Selection

Early Directional Commitment for Query-Efficient Black-Box Adversarial Attacks

[![arXiv](https://img.shields.io/badge/arXiv-2605.25663-b31b1b.svg)](https://arxiv.org/abs/2605.25663)
[![PDF](https://img.shields.io/badge/PDF-arXiv-red.svg)](https://arxiv.org/pdf/2605.25663)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2605.25663-blue.svg)](https://doi.org/10.48550/arXiv.2605.25663)
[![Venue](https://img.shields.io/badge/CAp-2026-2f855a.svg)](https://caprfiap2026.sciencesconf.org/page/program_cap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Opportunistic Target Selection (OTS)** is a lightweight wrapper for score-based black-box adversarial attacks that lack implicit target tracking. It runs the attack in untargeted mode for a short exploration phase, then switches to a targeted objective against whichever non-true class currently leads. OTS acts as a margin-loss surrogate: it reduces class drift for probability- or cross-entropy-based random-search attacks without requiring architectural modification, gradient access, or a priori target-class knowledge.

Across three score-based attacks and five standard ImageNet classifiers (4,500 runs), OTS closely tracks oracle targeting on drift-prone attacks, with gains up to +27 pp in success rate and 43% relative reduction in censored-mean iterations on ResNet-50. It is redundant for attacks that already provide directionality, such as Bandits or margin-loss Square Attack, and shows no benefit on adversarially-trained models where the difficulty distribution is bimodal.

This repository accompanies the CAp 2026 paper:

- Paper: [arXiv:2605.25663](https://arxiv.org/abs/2605.25663)
- PDF: [arxiv.org/pdf/2605.25663](https://arxiv.org/pdf/2605.25663)
- DOI: [10.48550/arXiv.2605.25663](https://doi.org/10.48550/arXiv.2605.25663)
- Venue: accepted for presentation at [CAp 2026](https://caprfiap2026.sciencesconf.org/page/program_cap), as listed in the official CAp program
- Source: [`paper/main.tex`](paper/main.tex)

---

## Project Structure

```
src/                    Core library
  attacks/              Attack implementations (SimBA, SquareAttack, Bandits)
  models/               Model loaders (torchvision, RobustBench)
  utils/                Image preprocessing & visualization
  demo/                 Gradio demonstrator app
demo/                   Demo entry point
benchmarks/             Benchmark scripts (generate CSV results)
analysis/               Analysis scripts (generate figures from CSVs)
slurm/                  HPC job scripts (CRIANN Arctic)
paper/                  LaTeX paper
results/                Benchmark CSVs and figures
data/                   ImageNet class index and demo images
```

---

## Quick Start

1. **Install dependencies**

    ```bash
    pip install -r requirements-gpu.txt   # With GPU (NVIDIA CUDA)
    pip install -r requirements-cpu.txt   # CPU only
    ```

2. **Launch the demonstrator**

    ```bash
    python demo/launch.py
    ```

3. **Access the interface**

    Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Benchmarks

Benchmark scripts expect ImageNet validation images in ImageFolder layout at `data/imagenet/val/`, or a symlink at that path. The ImageNet images used for the paper are not redistributed in this repository; keep a local copy under that path before rerunning the benchmarks.

| Script | Description |
|--------|-------------|
| `benchmarks/benchmark.py` | Multi-model benchmark: 5 standard models + 2 robust models, 3 attacks, 3 modes |
| `benchmarks/winrate.py` | ResNet-50 CDF benchmark: 100 images, 15K budget, bootstrapped CI |
| `benchmarks/ablation_s.py` | Stability threshold sweep S={2..15} on standard ResNet-50 |
| `benchmarks/ablation_s_robust.py` | Stability threshold sweep on robust ResNet-50 |
| `benchmarks/ablation_naive.py` | Naive fixed-iteration switching vs OTS (standard and robust) |
| `benchmarks/margin.py` | Margin vs CE loss ablation on SquareAttack |
| `benchmarks/landscape.py` | Per-iteration confidence history collection |
| `benchmarks/theta.py` | Perturbation alignment with oracle direction |

```bash
python benchmarks/benchmark.py
python benchmarks/winrate.py
```

## Analysis

Regenerate figures from benchmark CSVs:

```bash
python analysis/analyze_benchmark.py
python analysis/analyze_winrate.py
python analysis/analyze_ablation_s.py
python analysis/analyze_ablation_naive.py
python analysis/analyze_margin.py
python analysis/analyze_lockmatch.py
python analysis/analyze_oracle_beat.py
```

The benchmark CSVs and paper figures used for the arXiv version are included under `results/`.

## Citation

If you use this code, results, or figures, please cite:

```bibtex
@article{tariolle2026opportunistic,
  title = {Opportunistic Target Selection: Early Directional Commitment for Query-Efficient Black-Box Adversarial Attacks},
  author = {Tariolle, Florent and Yger, Florian},
  journal = {arXiv preprint arXiv:2605.25663},
  year = {2026},
  doi = {10.48550/arXiv.2605.25663},
  url = {https://arxiv.org/abs/2605.25663}
}
```

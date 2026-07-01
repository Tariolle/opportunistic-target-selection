# Opportunistic Target Selection
### Early Directional Commitment for Query-Efficient Black-Box Adversarial Attacks

[Florent Tariolle](https://tariolle.github.io/) and [Florian Yger](https://scholar.google.com/citations?user=NF_1_38AAAAJ)

**Abstract:** Black-box adversarial attacks that minimize only the ground-truth confidence suffer from class drift: perturbations wander through the feature space without committing to a specific adversarial class, wasting queries on diffuse, undirected progress. We introduce Opportunistic Target Selection (OTS), a lightweight wrapper that switches an untargeted attack to a targeted objective early in its trajectory, locking onto whichever non-true class currently leads. OTS requires no architectural modification to the underlying attack, no gradient access, and no a priori target-class knowledge.

<div align="center">
  <strong>[ <a href="https://arxiv.org/abs/2605.25663">Paper</a> | <a href="poster/poster_beamer.pdf">Poster</a> ]</strong><br>
  <strong>Accepted as a poster at <a href="https://caprfiap2026.sciencesconf.org/page/program_cap">CAp 2026</a></strong><br>
</div>

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

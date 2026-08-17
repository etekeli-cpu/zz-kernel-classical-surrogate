# Repository README (draft)

---

# Closed-form classical surrogate for the ZZ quantum feature map

Code and data for *"The ZZ feature map induces a signless-Laplacian metric: a closed-form classical surrogate for quantum kernel regression"* (Tekeli, 2026), submitted to *PeerJ Computer Science*.

We derive the metric induced by the ZZ feature map in the small-bandwidth
regime, $M = I + \pi^{2}Q$ with $Q$ the signless Laplacian of the
entanglement graph, and evaluate the corresponding parameter-free classical
kernel against the quantum kernel on two NIR spectroscopic regression
benchmarks.

## Requirements

```
python >= 3.10
numpy pandas scipy scikit-learn joblib matplotlib qiskit rdata
```

```bash
pip install -r requirements.txt
```

Datasets are downloaded automatically on first run from the public
repositories of the R packages `fda.usc` (Tecator) and `pls` (Gasoline).
No credentials or manual downloads are required.

## Layout

```
src/          analysis scripts
figures/      figure scripts
results/      pre-computed CSV outputs (see note on size)
paper/        manuscript source
```

## Reproducing the paper

Each script writes CSVs and prints the corresponding table. All support
`--n-jobs` and `--resume`; runtimes below are for 10 cores.

| Paper item | Command | Runtime |
|---|---|---|
| Table 1 (topology verification) | `python src/topology_test.py` | ~2 min |
| Table 2 (depth × convention) | `python src/reps_conv_test.py` | ~3 min |
| Tables 3, 5 · Figure 3 | `python src/s_threshold_full.py --n-reps 100 --n-jobs 10` | ~40 min |
| Table 4 (twin equivalence) | `python src/t3_twin_robustness.py --n-reps 100 --n-jobs 10` | ~30 min |
| Table 6 (dimension sweep) | `python src/pc_control_full.py --n-reps 100 --n-jobs 10 --with-quantum-anchor` | ~2 h |
| Table 7 (family comparison) | `python src/edf_study_v3.py --n-reps 100 --n-jobs 10 --qubits 6` | ~1.5 h |
| Figure 1 | `python figures/figure3.py --out fig1_metric --formats pdf` | seconds |
| Figure 2 | `python figures/figure2.py --out fig2_scatter --formats pdf` | ~1 min |
| Figure 3 | `python figures/figure1.py --by-target --out fig3_boundary --formats pdf` | seconds |

To regenerate a table from stored results without recomputing:

```bash
python src/s_threshold_full.py --report-only --outdir results/sthr
```

## Determinism

Train/test splits use `random_state = 1000 + replicate`, and
cross-validation folds use a fixed seed, so every family and preprocessing
pipeline sees identical partitions. Results are reproducible to floating
point across runs on the same platform.

## Notes on the analysis

- **Preprocessing is a hyperparameter.** Five pipelines (raw, SNV, first
  and second Savitzky–Golay derivatives, SNV+second derivative) are crossed
  with the kernel grid and selected by cross-validation. Fixing one pipeline
  favours some kernel families over others; see Section 2.3 of the paper.
- **Classical length-scale grids are anchored to the data**, spanning
  $\bar d\cdot 10^{[-1.5,1.5]}$ around the median pairwise distance. With a
  fixed grid the classical baselines select the grid boundary in every
  split, which changes which family attains the lowest error.
- **The twin kernel has no fitted parameters.** $M$ is read off the
  entanglement graph; only the bandwidth is selected, on the same grid used
  for the quantum kernel.

## Data

Analysis outputs are in `results/`. `s_threshold.csv` is ~10 MB; if hosting
constraints apply, regenerate it with the command above rather than storing
it, or place it on the Zenodo record only.

## Citation

```bibtex
@article{tekeli2026,
  author  = {Tekeli, Erkut},
  title   = {[TITLE]},
  journal = {PeerJ Computer Science},
  year    = {2026},
  doi     = {[DOI]}
}
```

## License

Code: MIT. Analysis outputs: CC BY 4.0.

---

# Gönderim öncesi kontrol listesi

- [ ] Temiz bir sanal ortamda `pip install -r requirements.txt` + her komutu
      **sıfırdan** koştur. Veri indirme, `--resume` mantığı ve import
      yolları çalışıyor mu?
- [ ] `t3_twin_robustness.py` ve `s_threshold_full.py`, `edf_study_v3.py`'yi
      `exec` ile okuyor ve **aynı dizinde** olmasını bekliyor. `src/`
      altında bu çalışır; farklı yerleştirirsen kırılır.
- [ ] README'deki tablo değerlerinden ikisini üçünü yeni koşuyla doğrula.
- [ ] `requirements.txt`'e sürüm sabitle (`qiskit==...`) — Qiskit API'si hızlı
      değişiyor ve `ZZFeatureMap` konvansiyonu makalenin merkezinde.
- [ ] Zenodo: GitHub deposunu bağla, sürüm etiketle (`v1.0`), DOI al, DOI'yi
      makalenin Data Availability bölümüne yaz.
- [ ] LICENSE dosyası ekle.

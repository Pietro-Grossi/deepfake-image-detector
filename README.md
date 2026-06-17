# GenImage AI Detector

Progetto di deep learning per il riconoscimento di immagini generate da intelligenza artificiale.

L'obiettivo è sviluppare e confrontare modelli di reti neurali in grado di distinguere immagini reali da immagini generate artificialmente, utilizzando il dataset GenImage.

## Obiettivi

- Preparare e analizzare il dataset GenImage.
- Addestrare un classificatore real/fake.
- Confrontare modelli basati su CNN e, opzionalmente, Transformer.
- Valutare la capacità del modello di generalizzare su immagini generate da modelli diversi.

## Tecnologie

- Python 3.11
- PyTorch / torchvision / timm
- scikit-learn, pandas
- matplotlib, seaborn
- Optuna (ottimizzazione iperparametri)
- TensorBoard (logging training)

## Struttura

```
configs/      # YAML per esperimenti (uno per modello)
data/         # (gitignored) subset GenImage + splits/ — solo sulla VM
scripts/      # download_subset, build_splits, make_fixtures, slurm/
src/          # datasets, transforms, degradations, models, train, evaluate, metrics, utils
tests/        # smoke test + fixtures finte per CPU
notebooks/    # EDA e visualizzazione risultati
experiments/  # (gitignored) output per-run: config, log, checkpoint
results/      # (gitignored) tabelle e grafici finali
```

## Workflow di sviluppo (locale ↔ VM)

Sviluppo **ibrido**: codice scritto/testato in locale (CPU), training eseguito sulla VM (GPU T4 via SLURM).
Il dataset e la GPU stanno **solo sulla VM**. Il ponte tra i due ambienti è **git**.

### Setup locale (CPU, per sviluppo e smoke test)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate su Linux/Mac
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Setup VM (GPU, CUDA)

```bash
conda env create -f environment.yml
conda activate genimage
```

### Smoke test (locale, senza GenImage)

```bash
python scripts/make_fixtures.py   # genera un dataset finto minuscolo in tests/fixtures/
pytest tests/test_smoke.py -v     # verifica utility di base su CPU
```

I componenti dati-dipendenti accettano `--data-root`, così lo stesso codice punta a
`tests/fixtures/` in locale e a `data/` sulla VM.

## Stato

- [x] **M0** — Setup ambiente, utility, fixtures, smoke test
- [ ] **M1** — Pipeline dati (subset GenImage, Dataset/DataLoader)
- [ ] **M2** — Baseline CNN + training/eval loop
- [ ] **M3** — Modelli pretrained (ResNet-50, EfficientNet-B0)
- [ ] **M4** — Transformer (Swin-Tiny)
- [ ] **M5** — Cross-generator · **M6** — Degradation · **M7** — Optuna · **M8** — Report
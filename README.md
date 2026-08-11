# MusicFusion

Official training code accompanying the MusicFusion paper on multi-view music
emotion recognition.

This repository is intended as a practical companion to the paper. The paper
contains the motivation, method description, and analysis; this page focuses on
preparing data, running the complete model, and reading its outputs.

## Installation

Python 3.8 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
git clone https://github.com/QilinLi147/MusicFusion.git
cd MusicFusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

The training entry point reads paired Mel-spectrogram and cochleagram arrays,
plus arousal and valence label arrays. The first dimension must refer to the
same samples in every file.

Place the prepared NumPy files under one data root using the layout below. You
only need the folder for the dataset you plan to run.

```text
<DATA_ROOT>/
├── memo/
│   ├── mel_spec.npy
│   ├── cochlegram.npy
│   └── labels/
│       ├── label_a.npy
│       └── label_v.npy
├── data_PMEmo/
│   ├── mel_spec.npy
│   ├── cochlegram.npy
│   ├── label_a.npy
│   └── label_v.npy
├── data_1000songs/
│   ├── mel_spec.npy
│   ├── cochlegram.npy
│   ├── label_a.npy
│   └── label_v.npy
└── DEAM/processed/
    ├── mel_spec.npy
    ├── cochlegram.npy
    └── labels/
        ├── label_a.npy
        └── label_v.npy
```

Dataset identifiers accepted by the command line are:

| Dataset | `--dataset` value |
| --- | --- |
| Memo2496 | `memo` |
| PMEmo | `pmemo` |
| 1000 Songs | `1000songs` |
| DEAM | `deam` |

Memo2496 can be obtained from
[Figshare](https://figshare.com/articles/dataset/Memo2496/25827034) or
[IEEE DataPort](https://dx.doi.org/10.21227/3824-wy49). Follow the respective
dataset licenses and access conditions for all datasets.

## Run the complete model

For Memo2496 arousal:

```bash
python train_musicfusion.py \
  --dataset memo \
  --label-mode a \
  --data-root /path/to/data \
  --run-dir runs/memo_arousal
```

For Memo2496 valence, change `--label-mode` to `v` and use a new output
directory:

```bash
python train_musicfusion.py \
  --dataset memo \
  --label-mode v \
  --data-root /path/to/data \
  --run-dir runs/memo_valence
```

The same command works for the other supported datasets by changing
`--dataset`. Each run directory must be empty or new.

To continue an interrupted run:

```bash
python train_musicfusion.py \
  --dataset memo \
  --label-mode a \
  --data-root /path/to/data \
  --run-dir runs/memo_arousal \
  --resume
```

Use `--device cpu` when CUDA is unavailable. Run
`python train_musicfusion.py --help` for the complete usage summary.

## Outputs

Training and validation are performed by the same command. The run directory
contains:

| File | Contents |
| --- | --- |
| `best.pt` | Best validation checkpoint |
| `last.pt` | Latest resumable checkpoint |
| `summary.json` | Final training and validation metrics |
| `history.jsonl` | Per-pass metrics and diagnostics |
| `split_indices.npz` | Train/validation indices |
| `split_manifest.json` | Split metadata |
| `config.json` | Reproducibility metadata for the completed run |

Accuracy, macro-F1, positive-class F1, precision, recall, and AUROC are written
to `summary.json` after evaluation.

## Memo2496 citation

If you use Memo2496, please cite:

> Q. Li, C. L. P. Chen and T. Zhang, "Memo2496: Expert-Annotated Dataset and
> Dual-View Adaptive Framework for Music Emotion Recognition," *IEEE
> Transactions on Affective Computing*, 2026, doi:
> [10.1109/TAFFC.2026.3715195](https://doi.org/10.1109/TAFFC.2026.3715195).

```bibtex
@article{li2026memo2496,
  author   = {Li, Q. and Chen, C. L. P. and Zhang, T.},
  title    = {Memo2496: Expert-Annotated Dataset and Dual-View Adaptive Framework for Music Emotion Recognition},
  journal  = {IEEE Transactions on Affective Computing},
  year     = {2026},
  doi      = {10.1109/TAFFC.2026.3715195},
  keywords = {Labeling; Music; Multiple signal classification; Modeling; Emotion recognition; Annotations; Convolutional neural networks; Tracking; Protocols; Conferences; Music emotion recognition; Affective computing; Dual-view learning; Cross-attention fusion; Pseudo-label learning; Contrastive memory; Expert-annotated dataset; Instrumental music dataset}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

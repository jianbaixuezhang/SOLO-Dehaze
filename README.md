# SOLO-Dehaze
SOLO-Dehaze: Scale-Oriented Laplacian Optimization with Contrastive Domain Alignment for Unsupervised Single Image Dehazing



## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset Configuration](#dataset-configuration)
- [Quick Start](#quick-start)
- [培训](#training)
- [Testing](#testing)
- [Weights and Checkpoints](#weights-and-checkpoints)
- [Evaluation Metrics](#evaluation-metrics)

---

## Project Structure

```
SOLO-Dehaze/
├── main.py                 # Entry point for training and testing
├── train.py                # Training, validation, and inference logic
├── test.py                 # Standalone test script
├── data/
│   └── dataset.py          # Dataset loaders (train / val / test)
├── models/
│   ├── SOLODehaze.py       # SOLO-Dehaze network
│   ├── TransformerBlock.py
│   ├── LightweightLinearAttn.py
│   ├── PhaseConsistencyLoss.py
│   ├── PerceptualLoss.py
│   ├── ContrastiveDomainLoss.py
│   └── ColorLoss.py
├── util/
│   └── metrics.py          # PSNR, SSIM, CIEDE2000, LPIPS
├── dataset/                # Place your data here (not included)
│   ├── train/
│   │   ├── hazy/
│   │   └── gt/
│   ├── val/
│   │   ├── hazy/
│   │   └── gt/
│   └── test/
│       ├── hazy/
│       └── gt/             # Optional; used only when GT is available
├── results/                # Created during training
│   ├── checkpoints/
│   ├── val_runs/
│   ├── loss_history.csv
│   └── loss_iteration.csv
└── Out/                    # Created during testing
```

---

## Environment Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

### Install dependencies

```bash
pip install torch torchvision numpy pillow einops
```

Optional packages for extended test metrics:

```bash
pip install scikit-image   # CIEDE2000
pip install lpips          # LPIPS
```

> **Note:** The perceptual loss uses a pretrained VGG-16 backbone (`torchvision.models.vgg16`), which is downloaded automatically on first run.

### Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Dataset Configuration

Place your dataset under the project root as `dataset/` (or pass a custom path via `--data_root`).

### Directory layout

```
dataset/
├── train/
│   ├── hazy/     # Hazy training images (unpaired)
│   └── gt/       # Clear training images (unpaired)
├── val/
│   ├── hazy/     # Hazy validation images
│   └── gt/       # Clear validation images (same filenames as hazy/)
└── test/
    ├── hazy/     # Hazy test images
    └── gt/       # Optional clear references (same filenames as hazy/)
```

### Supported image formats

`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`

### Split behavior

| Split | Pairing | Usage |
|-------|---------|-------|
| **train** | Unpaired | Hazy and clear images are sampled independently each iteration |
| **val** | Paired | Filenames must match between `val/hazy/` and `val/gt/` |
| **test** | Optional paired | Dehazing runs on all images in `test/hazy/`; metrics are computed only when a matching file exists in `test/gt/` |



---

## Quick Start

### 1. Prepare data

Organize images following the layout above and put them in `dataset/`.

### 2. Download pretrained weights (optional)

Pretrained weights and result files are available via Baidu Netdisk:

- **Share name:** SOLO_Dehaze  
- **Link:** https://pan.baidu.com/s/1yr3nKUG1pUmYYmHFwc1B9A  
- **Extract code:** `jjn3`

After downloading, place the weight file at:

```
results/checkpoints/best_weights.pt
```

### 3. Run inference

```bash
python main.py --mode test --data_root dataset --test_weights results/checkpoints/best_weights.pt
```

Or use the dedicated test script:

```bash
python test.py --data_root dataset --weights results/checkpoints/best_weights.pt --out Out
```

Dehazed images and metric reports will be saved under `Out/`.

---

##培训

### Basic command

```bash
python main.py --mode train --data_root dataset --results_dir results
```

Training starts from scratch by default. To disable auto-resume:

```bash
python main.py --mode train --no_resume
```

### Default hyperparameters

| Parameter | Default |描述|
|-----------|---------|-------------|
| `num_epochs` | 6000 | Total training epochs |
| `batch_size` | 4 | Training batch size |
| `patch_size` | 128 | Random crop size |
| `lr` | 1e-4 | Initial learning rate |
| `min_lr` | 1e-6 | Minimum learning rate (cosine schedule) |
| `warmup_epochs` | 200 | Linear warmup epochs |
| `val_interval` | 20 | Validate every N epochs |
| `log_interval` | 10 | Print loss every N iterations |



### Resume training

By default, training resumes from `results/checkpoints/last_full.pt` if it exists:

```bash
python main.py --mode train --resume_path results/checkpoints/last_full.pt
```

### Training outputs

During training, the following are written to `results/`:

| File / folder |描述|
|---------------|-------------|
| `checkpoints/last_full.pt` | Latest full checkpoint (model + optimizer + contrastive module) |
| `checkpoints/best_weights.pt` | Best dehazing weights (by validation score) |
| `checkpoints/best_full.pt` | Best full checkpoint |
| `val_runs/e{epoch}_psnr{..}_ssim{..}/` | Validation dehazed images and per-epoch metrics |
| `loss_history.csv` | Per-epoch loss summary |
| `loss_iteration.csv` | Per-iteration loss log |

Validation score used for model selection:

```
Score = PSNR + 100 × SSIM
```

---

## Testing

### Via main entry point

```bash
python main.py --mode test \
  --data_root dataset \
  --test_weights results/checkpoints/best_weights.pt \
  --test_out Out \
  --device cuda
```

### Via test script

```bash
python test.py \
  --data_root dataset \
  --weights results/checkpoints/best_weights.pt \
  --out Out \
  --device cuda \
  --num_workers 4
```

### Test outputs

Results are saved under `Out/` in a subfolder named by metrics, for example:

```
Out/test_psnr28.1234_ssim0.9123/
├── image_001.png              # Dehazed outputs
├── image_002.png
├── test_metrics_report.json   # Full metric report
└── per_image_psnr_ssim.csv    # Per-image metrics table
```

If no ground truth is available in `test/gt/`, only dehazed images are produced and the folder is timestamped instead.

---

## Weights and Checkpoints

### Download links

| Resource |地址|
|----------|----------|
| Pretrained weights & results | [Baidu Netdisk – SOLO_Dehaze](https://pan.baidu.com/s/1yr3nKUG1pUmYYmHFwc1B9A) (code: `jjn3`) |

### Checkpoint formats

| File | Contents | Use case |
|------|----------|----------|
| `best_weights.pt` | Dehazing network state dict only | Inference / testing |
| `last_full.pt` | Network + contrastive module + optimizer + config | Resume training |
| `best_full.pt` | Same as above, best validation epoch | Resume from best model |

### Loading weights for testing

```bash
python test.py --weights path/to/best_weights.pt
```

The model architecture must match the default configuration in `main.py` unless you override the `--dehaze_*` arguments consistently during both training and testing.

---

## Evaluation Metrics

All metrics are computed on full-resolution images (`eval: full_image_no_patch`).

| Metric |描述| Required package |
|--------|-------------|------------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | Built-in |
| **SSIM** | Structural Similarity Index | Built-in |
| **CIEDE2000** | Perceptual color difference | `scikit-image` |
| **LPIPS** | Learned perceptual image patch similarity | `lpips` |

### Validation metrics

Saved in each `results/val_runs/e{epoch}_.../` folder:

- `metrics.json` — mean PSNR, SSIM, and selection score
- `per_image_psnr_ssim.csv` — per-image PSNR and SSIM
- `weights.pt` — network weights at that epoch

### Test metrics

Saved in each `Out/test_psnr..._ssim.../` folder:

- `test_metrics_report.json` — per-image and average metrics
- `per_image_psnr_ssim.csv` — tabular export with PSNR, SSIM, CIEDE2000, LPIPS

---



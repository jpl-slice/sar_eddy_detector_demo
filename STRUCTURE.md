# Project Directory Structure

This document outlines the folder layout of the **SAR Eddy Detection Demo** project. It explains the purpose of each directory and helps you navigate the repository more easily.

## Directory Tree

```
config/
├── config.yaml                # (Optional) Base/global config entry
├── main_config.yaml           # Primary Hydra root config (loaded by main scripts)
├── inference.yaml             # Convenience composed config for standard inference
├── inference_timm_xgboost.yaml# Convenience config targeting timm+xgboost pipeline
├── hyp3/                      # Hydra config group: hyp3 job + granule selection
│   ├── default.yaml
│   └── granules.yaml
├── inference/                 # Hydra config group: inference variants
│   ├── default.yaml
│   └── timm_xgb.yaml
└── preprocessing/             # Hydra config group: preprocessing variants
    └── default.yaml

data/
├── land_mask/                 # Land/shoreline ancillary data (Natural Earth, etc.)
└── ...                        # Input SAR scenes & intermediate preprocessing outputs

model_checkpoints/
└── checkpoint.tar             # Pretrained checkpoint bundle (may include encoder + aux models)
└── eva02_large_patch14_448.mim_in22k_ft_in22k_xgboost_pipeline.pkl  # Pretrained timm+XGBoost model pipeline

src/
├── __init__.py
├── dataset.py                 # Dataset construction & tile generation
├── main.py                    # Standard entry point (Hydra-enabled)
├── transforms.py              # Image / tensor transform definitions used by dataset/model
├── visualize_eddy_bbox.py     # Visualization utilities for detected eddy bounding boxes
│
├── eddy_detector/             # Detector abstraction & concrete detector implementations
│   ├── __init__.py
│   ├── base_detector.py       # Abstract / shared detection logic
│   ├── simclr_detector.py     # Detector using SimCLR feature backbone
│   └── timm_xgboost_detector.py # Detector using timm backbone + XGBoost classifier
│
├── models/                    # Model & backbone loaders (feature extraction architectures)
│   ├── __init__.py
│   ├── simclr_loader.py       # Load SimCLR checkpoints / weights
│   ├── simclr_resnet.py       # SimCLR ResNet backbone definition
│   └── timm_loader.py         # timm model loader / wrapper
│
├── sar_utils/                 # SAR-specific preprocessing & utilities
│   ├── __init__.py
│   ├── download_rtc_from_hyp3.py        # HyP3 API download helper
│   ├── plotting.py                        # Plotting helpers
│   ├── preprocess_land_mask_and_normalize.py # Land mask + normalization pipeline
│   └── transforms.py                     # SAR-domain specific transforms
│
└── utils/                     # General utility helpers
    ├── __init__.py
    ├── bbox.py                # Bounding box manipulation functions
    ├── compress_sar_with_jpeg2000.py # Optional JP2 compression routines
    ├── config.py              # Hydra config loading / composition helpers
    ├── file_preprocess_checker.py # File readiness / preprocessing state checks
    ├── importer.py            # Dynamic import utilities
    └── raster_io.py           # Raster read/write utilities

output/                        # Primary output directory (detections, previews, logs); can customize output directory via CLI
```

## Folder Descriptions

- **config/**  
  Hydra-based configuration system. Top-level YAML files (e.g., `main_config.yaml`) are composed configs for common runs. Subdirectories (`hyp3/`, `inference/`, `preprocessing/`) are Hydra config groups; you can override parts of a run via CLI, e.g.:  
  `python -m src.main hyp3=granules inference=timm_xgb preprocessing=default`  
  or supply ad‑hoc overrides:  
  `python -m src.main inference.threshold=0.9 output_dir=output/experiment_01`  
  The system enables hierarchical, modular configuration without editing code.

- **data/**  
  Raw & intermediate SAR imagery plus ancillary data (land mask, etc.). Preprocessing writes normalized / tiled products into subfolders before detection runs.

- **model_checkpoints/**  
  Bundled pretrained model artifacts (encoders, detector heads, auxiliary components) loaded at inference time.

- **src/**  
  All source code: entry (`main.py`), dataset & transform logic, detector abstractions, backbone loaders, SAR domain utilities, and general helpers.

  - `eddy_detector/`: Implements the detector interface and concrete strategies (SimCLR, timm+XGBoost).  
  - `models/`: Backbones & loaders for feature extraction.  
  - `sar_utils/`: Domain-specific preprocessing (HyP3 downloads, masking, normalization).  
  - `utils/`: Generic utilities (config integration, IO, bbox operations, compression, dynamic imports).  

- **output/** / **outputs/** / **output_test/**  
  Generated artifacts: detection CSVs, preview imagery, logs, date-stamped batch outputs, and experimental/test runs.

## Usage

See the [Installation Guide](INSTALLATION.md) for environment setup and running the pipeline. This document focuses on repository layout and configuration structure.

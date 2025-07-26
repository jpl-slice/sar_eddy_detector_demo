#!/usr/bin/env python3
"""
Main orchestrator for downloading, preprocessing, and running inference on SAR data from ASF HYP3.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf

# Add repo root to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sar_utils import build_land_masker, initialize_hyp3_client
from src.sar_utils.download_rtc_from_hyp3 import fetch_tif_from_hyp3
from src.sar_utils.preprocess_land_mask_and_normalize import preprocess_frame


def parse_config(config_path: str):
    """A simple YAML config loader."""
    return OmegaConf.load(config_path)


def main():
    """Main workflow orchestrator."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/main_config.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    cfg = parse_config(args.config)

    setup_logging(cfg.paths.log_dir)

    hyp3_client = initialize_hyp3_client(cfg.hyp3.username, cfg.hyp3.password)
    land_masker = build_land_masker(cfg.preprocessing.land_shapefile)

    for granule in cfg.workflow.granules:
        run_granule_workflow(granule, hyp3_client, land_masker, cfg)


def run_granule_workflow(granule, hyp3_client, land_masker, cfg):
    """Run the complete download and preprocess workflow for a single granule."""
    logging.info(f"Starting workflow for granule: {granule}")
    try:
        download_dir, temp_dir = Path(cfg.paths.download_dir), Path(cfg.paths.temp_dir)
        raw_tif = fetch_tif_from_hyp3(
            hyp3_client, granule, download_dir, temp_dir, cfg.workflow.keep_zip
        )
        if not raw_tif:
            logging.error(f"No TIFF file for granule {granule}.")
            return

        # Preprocess & preview in one call
        suffix = cfg.preprocessing.processed_suffix
        out_tif = Path(cfg.preprocessing.processed_dir) / f"{raw_tif.stem}{suffix}.tif"
        preview_png = (
            Path(cfg.preprocessing.preview_dir) / f"{raw_tif.stem}_preview.png"
        )
        preprocess_frame(raw_tif, out_tif, preview_png, land_masker, cfg.preprocessing)

        logging.info(f"Saved processed file: {out_tif} and preview: {preview_png}")
        logging.info(f"Successfully processed granule: {granule}")

    except Exception as e:
        logging.exception(f"Error processing granule {granule}: {e}")


def setup_logging(log_dir: str):
    """Configure logging for the application."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path / "run_workflow.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


if __name__ == "__main__":
    main()

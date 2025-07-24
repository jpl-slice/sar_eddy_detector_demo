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

from src.sar_utils import (
    build_land_masker,
    download_and_extract_files,
    get_or_submit_rtc_job,
    initialize_hyp3_client,
    mask_land_and_clip,
    monitor_job_completion,
    quicklook,
    write_masked,
)


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
    land_masker = build_land_masker(cfg.preprocess.land_shapefile)

    for granule in cfg.workflow.granules:
        run_granule_workflow(granule, hyp3_client, land_masker, cfg)


def run_granule_workflow(granule, hyp3_client, land_masker, cfg):
    """Run the complete download and preprocess workflow for a single granule."""
    logging.info(f"Starting workflow for granule: {granule}")
    try:
        # Download
        job = get_or_submit_rtc_job(hyp3_client, granule, dict(cfg.job_parameters))
        if not job:
            return
        completed_job = monitor_job_completion(hyp3_client, job)
        if not completed_job or not completed_job.succeeded():
            logging.error(f"Job for {granule} failed or was not completed.")
            return

        download_dir = Path(cfg.paths.download_dir)
        temp_dir = Path(cfg.paths.temp_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = download_and_extract_files(
            completed_job, download_dir, temp_dir, cfg.workflow.keep_zip
        )
        if not downloaded_files:
            logging.error(f"No files downloaded for granule {granule}.")
            return

        raw_tif = [f for f in downloaded_files if f.suffix == ".tif"][0]

        # Preprocess
        processed_dir = Path(cfg.paths.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = Path(cfg.paths.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        out_tif = processed_dir / f"{raw_tif.stem}_processed.tif"

        with rasterio.open(raw_tif) as src:
            # The land_masker is now passed directly, consistent with the
            # updated `mask_land_and_clip` function.
            masked = mask_land_and_clip(
                src,
                land_masker,
                clip_percentile=cfg.preprocess.clip_percentile,
                dilate_px=cfg.preprocess.dilate_px,
            )

            if cfg.preprocess.convert_to_db:
                epsilon = 1e-10
                valid = np.isfinite(masked)
                masked[valid] = 10 * np.log10(masked[valid] + epsilon)
                masked = np.clip(masked, -35, None)

            if cfg.preprocess.save_masked:
                write_masked(
                    src,
                    masked,
                    out_tif,
                    cfg.preprocess.masked_dtype,
                    compress=cfg.preprocess.compress,
                )
                logging.info(f"Saved processed file: {out_tif}")

            quicklook_png = preview_dir / f"{raw_tif.stem}_preview.png"
            quicklook(src, masked, quicklook_png)
            logging.info(f"Saved preview: {quicklook_png}")

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

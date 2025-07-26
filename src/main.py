#!/usr/bin/env python3
"""
Unified orchestrator for SAR eddy detection using Hydra configuration management.
Supports full workflow (HyP3 download + preprocessing + inference) or individual stages.
"""
import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add repo root to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sar_utils import build_land_masker, initialize_hyp3_client
from src.sar_utils.download_rtc_from_hyp3 import fetch_tif_from_hyp3
from src.sar_utils.preprocess_land_mask_and_normalize import preprocess_frame
from src.utils import load_class


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main workflow orchestrator supporting multiple modes."""
    setup_logging(cfg)

    if cfg.mode in ["full", "hyp3_only"]:
        run_hyp3_workflow(cfg)

    if cfg.mode in ["full", "inference_only"]:
        # TODO: perhaps only run inference on granules processed in the HyP3 workflow
        run_inference_workflow(cfg)

    logging.info("Workflow completed successfully!")


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging for the application."""
    log_path = Path(cfg.paths.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path / "sar_eddy_detector.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_hyp3_workflow(cfg: DictConfig) -> None:
    """Run HyP3 download and preprocessing workflow."""
    logging.info("Starting HyP3 download and preprocessing workflow")

    hyp3_client = initialize_hyp3_client(cfg.hyp3.username, cfg.hyp3.password)
    land_masker = build_land_masker(cfg.preprocessing.land_shapefile)

    for granule in cfg.hyp3.granules:
        run_granule_workflow(granule, hyp3_client, land_masker, cfg)


def run_granule_workflow(
    granule: str, hyp3_client, land_masker, cfg: DictConfig
) -> None:
    """Run the complete download and preprocess workflow for a single granule."""
    logging.info(f"Starting workflow for granule: {granule}")
    try:
        download_dir = Path(cfg.paths.download_dir)
        temp_dir = Path(cfg.paths.temp_dir)

        raw_tif = fetch_tif_from_hyp3(
            hyp3_client,
            granule,
            download_dir,
            temp_dir,
            cfg.hyp3.keep_zip,
            dict(cfg.hyp3.job_parameters) if cfg.hyp3.job_parameters else None,
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


def run_inference_workflow(cfg: DictConfig) -> None:
    """Run inference workflow for eddy detection."""
    logging.info("Starting inference workflow")

    # Ensure inference config has necessary paths after potential HyP3 workflow
    if cfg.mode == "full":
        # In full mode, use processed_dir as the default geotiff_dir for inference
        cfg.inference.geotiff_dir = cfg.preprocessing.processed_dir

    detector = initialize_detector(cfg.inference)
    if detector.setup():
        detector.run_inference()
        output_dir = cfg.inference.output_dir
        logging.info(f"Inference complete. Results saved in: {output_dir}")

        # Create overall previews
        detector.create_scene_previews_with_bbox(confidence_threshold=0.5, merged=True)

        # Create individual previews for each detection
        detector.save_positive_detection_tiles(
            confidence_threshold=0.5, merged=False, patch_size=256
        )


def initialize_detector(inf_cfg: DictConfig):
    """Dynamically imports and initializes the detector class based on inference config."""
    detector_class = inf_cfg.detector_class_name
    try:
        DetectorClass = load_class(detector_class, default_pkg="src.eddy_detector")
        detector = DetectorClass(inf_cfg)
        logging.info(f"Successfully instantiated detector: {detector_class}")
    except (AttributeError, ImportError) as e:
        logging.error(
            f"Error: Could not find or import the detector class '{detector_class}'. Check config."
        )
        logging.error(f"Details: {e}")
        sys.exit(1)
    return detector


if __name__ == "__main__":
    main()

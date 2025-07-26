#!/usr/bin/env python3
"""
Production script for processing Sentinel-1 SAR frames using HyP3 RTC processing.
Handles job submission, monitoring, downloading, and file extraction with credit management.

This module is designed to work with Hydra configuration management.
"""

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import hyp3_sdk


def initialize_hyp3_client(
    username: Optional[str], password: Optional[str]
) -> hyp3_sdk.HyP3:
    """Initialize HyP3 client with authentication."""
    if username and password:
        return hyp3_sdk.HyP3(username=username, password=password)

    try:
        return hyp3_sdk.HyP3()  # Try .netrc first
    except Exception:
        logging.info(".netrc authentication failed, prompting for credentials")
        return hyp3_sdk.HyP3(prompt="password")  # Prompt as last resort


def display_account_statistics(client: hyp3_sdk.HyP3):
    """Display account information and recent job statistics."""
    try:
        credits = client.check_credits()
        costs = client.costs()
        rtc_cost = costs.get("RTC_GAMMA", {}).get("cost", "Unknown")

        logging.info(f"Remaining credits: {credits}")
        logging.info(f"RTC job cost: {rtc_cost} credits each")

        recent_jobs = client.find_jobs()
        if recent_jobs:
            logging.info(f"Total jobs found: {len(recent_jobs)}")
            status_counts = recent_jobs._count_statuses()
            for status, count in status_counts.items():
                logging.info(f"  {status}: {count}")

            total_cost = recent_jobs.total_credit_cost()
            logging.info(f"Total credit cost of jobs: {total_cost}")

    except Exception as e:
        logging.warning(f"Could not retrieve account statistics: {e}")


def fetch_tif_from_hyp3(
    client: hyp3_sdk.HyP3,
    granule: str,
    output_dir: Path,
    temp_dir: Path,
    keep_zip: bool,
    job_parameters: Optional[dict] = None,
) -> Optional[Path]:
    """Process a single granule through the complete RTC workflow."""
    logging.info(f"Processing granule: {granule}")
    try:
        job = get_or_submit_rtc_job(client, granule, job_parameters)
        if not job:
            return None

        completed_job = monitor_job_completion(client, job)
        if not completed_job or not completed_job.succeeded():
            logging.error(f"Job for {granule} failed or was not completed.")
            return None

        downloaded_files = download_and_extract_files(
            completed_job, output_dir, temp_dir, keep_zip
        )
        if not downloaded_files:
            logging.error(f"No files downloaded for granule {granule}.")
            return None

        tif_files = [f for f in downloaded_files if f.suffix.lower() == ".tif"]
        if not tif_files:
            logging.error(f"No .tif file found for granule {granule}.")
            return None

        logging.info(f"Successfully processed granule: {granule}")
        return tif_files[0]

    except Exception as e:
        log_job_failure(granule, str(e))
        return None


def get_or_submit_rtc_job(
    client: hyp3_sdk.HyP3, granule: str, job_parameters: Optional[dict] = None
) -> Optional[hyp3_sdk.Job]:
    """Check for existing job or submit new RTC job for granule."""
    if job_parameters is None:
        job_parameters = dict(
            dem_name="copernicus",
            resolution=30,
            radiometry="sigma0",
            scale="power",
            speckle_filter=True,
            dem_matching=False,
            include_dem=False,
            include_inc_map=False,
            include_rgb=False,
            include_scattering_area=False,
        )
    existing_job = find_existing_job(client, granule, job_parameters)
    if existing_job:
        logging.info(f"Found existing job for granule {granule}: {existing_job.job_id}")
        return existing_job

    logging.info(f"Submitting new RTC job for granule: {granule}")
    batch = client.submit_rtc_job(
        granule=granule, name=f"RTC_{granule[-8:]}", **job_parameters
    )
    # Handle batch or single job return
    return batch if hasattr(batch, "job_id") else (batch[0] if batch else None)


def find_existing_job(
    client: hyp3_sdk.HyP3, granule: str, job_params: dict
) -> Optional[hyp3_sdk.Job]:
    """Find a non-failed, unexpired job for a given granule."""
    jobs = client.find_jobs(job_type="RTC_GAMMA")
    for job in jobs:
        job_parameters = getattr(job, "job_parameters", {}) or {}
        if (
            job_parameters.get("granules") == [granule]
            and not job.failed()
            and not job.expired()
            and job_parameters_match(job, job_params)
        ):
            return job
    return None


def job_parameters_match(job: hyp3_sdk.Job, current_params: dict) -> bool:
    """Check if job parameters match the current parameters."""
    job_params = job.job_parameters or {}
    for key, value in current_params.items():
        if job_params.get(key) != value:
            return False
    return True


def monitor_job_completion(
    client: hyp3_sdk.HyP3, job: hyp3_sdk.Job
) -> Optional[hyp3_sdk.Job]:
    """Monitor job until completion with timeout."""
    if job.complete():
        return job

    logging.info(f"Waiting for job {job.job_id} to complete...")
    try:
        result = client.watch(job, timeout=3600, interval=60)
        return job  # Return the original job since watch updates it
    except Exception as e:
        logging.error(f"Job monitoring timeout: {e}")
        return None


def download_and_extract_files(
    job: hyp3_sdk.Job, output_dir: Path, temp_dir: Path, keep_zip: bool
) -> list[Path]:
    """Download job files and extract specific file types."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    full_granule_name = get_granule_name_from_job(job)

    # if target TIFF already present, skip download/extract
    expected_tif = output_dir / f"{full_granule_name}.tif"
    if expected_tif.exists():
        logging.info(f"TIF file already exists: {expected_tif}, skipping download.")
        return [expected_tif]

    downloaded_files = job.download_files(temp_dir)
    extracted_files = []

    for zip_path in downloaded_files:
        if zip_path.suffix.lower() == ".zip":
            extracted_files.extend(
                extract_target_files(zip_path, full_granule_name, output_dir)
            )
            if not keep_zip:
                zip_path.unlink()
                logging.info(f"Deleted zip file: {zip_path}")
    return extracted_files


def get_granule_name_from_job(job: hyp3_sdk.Job) -> str:
    """Extract the full granule name from job parameters."""
    if job.job_parameters and "granules" in job.job_parameters:
        return job.job_parameters["granules"][0]
    return "unknown_granule"


def extract_target_files(
    zip_path: Path, full_granule_name: str, output_dir: Path
) -> list[Path]:
    """Extract VV polarization TIF and non-RGB PNG files from zip with renamed filenames."""
    tif_dir = output_dir
    png_dir = output_dir / "png"
    tif_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for file_info in zip_file.infolist():
            filename = file_info.filename

            if should_extract_file(filename):
                new_filename = generate_new_filename(filename, full_granule_name)
                target_dir = tif_dir if filename.endswith(".tif") else png_dir
                extract_path = target_dir / new_filename
                with zip_file.open(file_info) as source, open(
                    extract_path, "wb"
                ) as target:
                    target.write(source.read())

                logging.info(f"Extracted: {extract_path}")
                extracted_paths.append(extract_path)
    return extracted_paths


def should_extract_file(filename: str) -> bool:
    """Determine if file should be extracted based on naming criteria."""
    filename_lower = filename.lower()

    if filename_lower.endswith("_vv.tif"):
        return True

    if filename_lower.endswith(".png") and "rgb" not in filename_lower:
        return True

    return False


def generate_new_filename(original_filename: str, full_granule_name: str) -> str:
    """Generate new filename using full granule name instead of short name."""
    original_path = Path(original_filename)
    extension = original_path.suffix

    if extension.lower() == ".tif":
        return f"{full_granule_name}{extension}"
    elif extension.lower() == ".png":
        return f"{full_granule_name}_browse{extension}"

    return original_path.name


def log_job_failure(granule: str, error_message: str):
    """Log job failure with granule and error details."""
    msg = f"GRANULE: {granule} | ERROR: {error_message}"
    logging.getLogger("failed_jobs").error(msg)
    logging.error(f"Failed to process granule {granule}. See failed_jobs.log.")

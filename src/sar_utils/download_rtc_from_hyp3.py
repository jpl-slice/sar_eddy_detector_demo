#!/usr/bin/env python3
"""
Production script for processing Sentinel-1 SAR frames using HyP3 RTC processing.
Handles job submission, monitoring, downloading, and file extraction with credit management.

# Basic usage
python download_rtc_from_hyp3.py S1A_IW_SLC__1SDV_20240101T123456_20240101T123500_012345_012345_1234

# Multiple granules with custom output directory
python download_rtc_from_hyp3.py granule1 granule2 -o /path/to/output --keep-zip

# With credentials
python download_rtc_from_hyp3.py granule1 -u username -p password

You can also store your credentials in a ~/.netrc file, e.g.:

machine urs.earthdata.nasa.gov
login YOUR_USERNAME
password YOUR_PASSWORD
"""

import argparse
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import hyp3_sdk


def main():
    """Main entry point for RTC processing workflow."""
    args = parse_arguments()
    setup_logging(args.log_dir)

    hyp3_client = initialize_hyp3_client(args.username, args.password)
    display_account_statistics(hyp3_client)

    for granule in args.granules:
        process_granule(
            hyp3_client, granule, args.output_dir, args.log_dir, args.keep_zip
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Sentinel-1 SAR frames with HyP3 RTC"
    )
    parser.add_argument(
        "granules", nargs="+", help="Sentinel-1 granule names to process"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory",
    )
    parser.add_argument(
        "-l", "--log-dir", type=Path, default=Path("./logs"), help="Log directory"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="Keep downloaded zip files"
    )
    parser.add_argument("-u", "--username", help="NASA Earthdata username")
    parser.add_argument("-p", "--password", help="NASA Earthdata password")
    return parser.parse_args()


def setup_logging(log_dir: Path):
    """Configure logging for the application."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "hyp3_processor.log"),
            logging.StreamHandler(),
        ],
    )

    error_handler = logging.FileHandler(log_dir / "failed_jobs.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    error_logger = logging.getLogger("failed_jobs")
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)


def initialize_hyp3_client(
    username: Optional[str], password: Optional[str]
) -> hyp3_sdk.HyP3:
    """Initialize HyP3 client with authentication."""
    if username and password:
        return hyp3_sdk.HyP3(username=username, password=password)

    try:
        return hyp3_sdk.HyP3()  # Try .netrc first
    except hyp3_sdk.exceptions.AuthenticationError:
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


def process_granule(
    client: hyp3_sdk.HyP3, granule: str, output_dir: Path, log_dir: Path, keep_zip: bool
):
    """Process a single granule through the complete RTC workflow."""
    logging.info(f"Processing granule: {granule}")

    try:
        job = get_or_submit_rtc_job(client, granule)
        if not job:
            return

        completed_job = monitor_job_completion(client, job)
        if not completed_job or not completed_job.succeeded():
            log_job_failure(granule, "Job failed to complete successfully")
            return

        download_and_extract_files(completed_job, output_dir, keep_zip)
        logging.info(f"Successfully processed granule: {granule}")

    except Exception as e:
        log_job_failure(granule, str(e))


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
    return batch[0]


def find_existing_job(
    client: hyp3_sdk.HyP3, granule: str, job_params: dict
) -> Optional[hyp3_sdk.Job]:
    """Find a non-failed, unexpired job for a given granule."""
    jobs = client.find_jobs(job_type="RTC_GAMMA")
    for job in jobs:
        if (
            job.job_parameters.get("granules") == [granule]
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
        return client.watch(job, timeout=3600, interval=60)
    except hyp3_sdk.exceptions.HyP3Error as e:
        logging.error(f"Job monitoring timeout: {e}")
        return None


def download_and_extract_files(
    job: hyp3_sdk.Job, output_dir: Path, temp_dir: Path, keep_zip: bool
) -> list[Path]:
    """Download job files and extract specific file types."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    full_granule_name = get_granule_name_from_job(job)
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


if __name__ == "__main__":
    main()

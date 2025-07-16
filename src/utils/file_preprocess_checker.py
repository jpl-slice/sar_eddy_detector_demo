from pathlib import Path
from typing import Union

import rasterio
from rasterio.windows import Window


class PreprocessingError(Exception):
    """Raised when a file is not properly preprocessed."""

    pass


def check_file_is_preprocessed(
    geotiff_input: Union[str, Path], require_preprocessing: bool = True
) -> bool:
    """Check if a GeoTIFF file has been preprocessed (land-masked and normalized).

    This function uses multiple heuristics to determine if a SAR GeoTIFF has gone through
    a preprocessing pipeline, based on analysis of processed vs unprocessed file patterns.

    Detection Rules (in order of reliability):
    ----------------------------------------

    1. **Explicit Tags**: Look for custom preprocessing tags like 'preprocessed',
       'land_masked', 'normalized' in file metadata.

    2. **Compression**: Processed files typically use:
       - Some form of compression (e.g., DEFLATE, JPEG2000)
       - This is a strong indicator of preprocessing pipeline output

    3. **Tiling Structure**: Processed files consistently use:
       - 512x512 pixel tiles (blockxsize=512, blockysize=512)
       - Unprocessed files often use strip-based or full-image blocks

    4. **Coordinate Reference System**:
       - Processed files are reprojected to Geographic WGS84 (EPSG:4326)
       - Unprocessed files often remain in projected coordinates (e.g., UTM)

    5. **File Structure Indicators**:
       - COG layout or built-in overviews typically indicate unprocessed files
       - Processed files generally don't have overviews

    6. **Data Type and Range**:
       - Byte (uint8) data type strongly indicates processed files
       - Float32 with normalized value ranges suggests processing
       - Float32 with wide/raw ranges suggests unprocessed data

    7. **Metadata Signatures**:
       - Processing software tags (TIFFTAG_SOFTWARE, TIFFTAG_IMAGEDESCRIPTION)
         often indicate unprocessed or intermediate products
       - Clean metadata suggests processed files

    Args:
        filepath: Path to the GeoTIFF file to check
        require_preprocessing: If True, raise PreprocessingError for unprocessed files

    Returns:
        True if file appears preprocessed, False otherwise

    Raises:
        PreprocessingError: If require_preprocessing=True and file not preprocessed

    Example:
        >>> check_file_is_preprocessed("processed_image.tif")
        True
        >>> check_file_is_preprocessed("raw_sar.tif", require_preprocessing=False)
        False
    """

    try:
        with rasterio.open(geotiff_input) as src:
            score = 0
            # Run all checks and aggregate score
            score += _check_tags(src.tags())
            if score >= 10:  # Early exit if explicit tag was found
                return True

            score += _check_profile_and_structure(src)
            score += _check_crs(src)
            score += _check_data_type_and_range(src)

            is_preprocessed = score > 0

            if require_preprocessing and not is_preprocessed:
                raise PreprocessingError(
                    f"File {geotiff_input} does not appear to be preprocessed (score: {score}).\n"
                    "Please run preprocessing pipeline first."
                )

            return is_preprocessed

    except rasterio.errors.RasterioIOError as e:
        if require_preprocessing:
            raise PreprocessingError(f"Cannot read file {geotiff_input}: {e}")
        return False


def _check_tags(tags):
    """Rule 1 & 7: Check for explicit and software-related metadata tags."""
    score = 0
    # Rule 1: Definitive check for explicit 'processed' tag
    if any(
        tag in tags
        for tag in ["preprocessed", "land_masked", "normalized", "processed"]
    ):
        return 10  # Return a high score to exit early

    # Rule 7: Check for unprocessed software metadata
    software_tags = ["TIFFTAG_SOFTWARE", "TIFFTAG_IMAGEDESCRIPTION", "TIFFTAG_DATETIME"]
    if any(tag in tags for tag in software_tags):
        score -= 1
    # Also check tag values for common SAR software keywords
    tag_values = " ".join(str(v) for v in tags.values()).lower()
    if any(sig in tag_values for sig in ["gamma", "snap", "esa", "hyp3"]):
        score -= 2
    return score


def _check_profile_and_structure(src):
    """Rule 2, 3, 5: Check profile for compression, tiling, and COG structure."""
    score = 0
    prof = src.profile

    # Rule 2: Check for any compression
    if prof.get("compress") not in (None, "NONE", ""):
        score += 2
    # We don't penalize lack of compression, because a file can be land masked without compression
    # else:
    #     score -= 1  # Lack of compression is an unprocessed indicator

    # Rule 3: Check for 512x512 tiling
    # note that some HYP3 downloads use a blocksize of 256x256, so we don't reward/penalize that
    if prof.get("blockxsize") == 512 and prof.get("blockysize") == 512:
        score += 2
    else:
        # strip‐tiles
        if prof.get("blockysize", 0) == 1:
            score -= 1
        # full‐image blocks
        if (
            prof.get("blockxsize", 0) >= src.width
            and prof.get("blockysize", 0) >= src.height
        ):
            score -= 1

    # Rule 5: Check for COG layout or overviews
    is_cog = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
    has_overviews = any(src.overviews(band) for band in src.indexes)
    if is_cog or has_overviews:
        score -= 2
    return score


def _check_crs(src):
    """Rule 4: Check Coordinate Reference System."""
    if src.crs is None:
        return 0
    try:
        return 1 if src.crs.is_geographic else -1
    except AttributeError:
        return 0  # Unexpected CRS object type


def _check_data_type_and_range(src):
    """Rule 6: Check band data type and value range."""
    dtype = src.dtypes[0]
    if "int" in dtype.lower():
        return 2  # Very strong indicator

    score = 0
    if "float" in dtype.lower():
        sample = src.read(
            1,
            window=Window(0, 0, min(256, src.width), min(256, src.height)),
            boundless=True,
            fill_value=src.nodata or 0,
        )
        valid_data = sample[sample != (src.nodata or 0)]

        if valid_data.size > 0:
            min_val, max_val = float(valid_data.min()), float(valid_data.max())
            is_normalized = (0 <= min_val and max_val <= 1.1) or (
                0 <= min_val and max_val <= 255.1
            )
            is_wide_range = max_val > 1000 or min_val < -1000
            if is_normalized:
                score += 1
            elif is_wide_range:
                score -= 1
    return score


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_preprocess_checker.py <filepath>")
        sys.exit(1)
    filepath = sys.argv[1]
    try:
        is_preprocessed = check_file_is_preprocessed(filepath)
        print(
            f"File {filepath} is {'preprocessed' if is_preprocessed else 'not preprocessed'}."
        )
    except PreprocessingError as e:
        print(f"Error: {e}")
        sys.exit(1)

"""
This module provides utility functions for reading raster data, abstracting away
the specifics of different file formats like GeoTIFF and JPEG2000.
"""

from typing import Any, Union

import numpy as np
import rasterio
from rasterio import windows

from src.utils.compress_sar_with_jpeg2000 import expand_from_jp2


def get_nodata_from_src(src: rasterio.DatasetReader) -> float:
    """Determines the nodata value for a rasterio dataset.

    This function first attempts to read the nodata value directly from the
    raster's metadata (`src.nodata`). If the metadata value is not set (is None),
    it falls back to a heuristic to infer the nodata value, which is common
    for some SAR data products.

    The heuristic works as follows:
    1. A small sample (up to 128x128 pixels) is read from the upper-left
       corner of the raster.
    2. If any pixel in this sample has a value less than -9000, the function
       assumes a conventional SAR nodata value of -9999.0.
    3. Otherwise, it defaults to 0.0.

    Args:
        src: An open rasterio.DatasetReader object.

    Returns:
        The determined nodata value as a float.
    """
    # return self.src.nodata if self.src.nodata is not None else -9999.0
    if src.nodata is not None:
        return src.nodata
    else:
        # Heuristic sample (≤128×128) from UL corner
        data = src.read(
            1,
            window=windows.Window(0, 0, min(128, src.width), min(128, src.height)),
            boundless=True,
            fill_value=0,
        )  # Read a sample
        if np.any(data < -9000):
            # If nodata is not set, assume a common nodata value for SAR data
            return -9999.0
        else:
            return 0.0  # Default to 0 if not otherwise determined.


def read_raster_data(
    src_or_file: Union[rasterio.DatasetReader, str], **kwargs: Any
) -> np.ndarray:
    """
    Reads data from a raster source, dispatching to specialized handlers if necessary.
    Note that nodata pixels (if they exist) will be returned as NaN.
    """
    src = rasterio.open(src_or_file) if isinstance(src_or_file, str) else src_or_file
    is_jp2 = src.driver == "JP2OpenJPEG" or src.name.lower().endswith(".jp2")
    if is_jp2:
        data = expand_from_jp2(src.name, **kwargs).squeeze()
    else:
        data = src.read(1, **kwargs)

    nodata_value = get_nodata_from_src(src)
    if nodata_value is not None:
        data[data == nodata_value] = np.nan

    return data.squeeze()
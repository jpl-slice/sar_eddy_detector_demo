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
    # return self.src.nodata if self.src.nodata is not None else -9999.0
    if src.nodata is not None:
        return src.nodata
    else:
        data = src.read(
            1,
            window=windows.Window(0, 0, min(128, src.width), min(128, src.height)),
            boundless=True,
            fill_value=0,
        )  # Read a sample
        if np.any(data < -9000):  # Heuristic from user
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
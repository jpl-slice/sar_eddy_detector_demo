"""
This module provides utility functions for reading raster data, abstracting away
the specifics of different file formats like GeoTIFF and JPEG2000.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.utils.compress_sar_with_jpeg2000 import expand_from_jp2


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
        return expand_from_jp2(src.name, **kwargs).squeeze()
    return src.read(1, **kwargs).squeeze()
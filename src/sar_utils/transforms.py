"""
Land masking, nodata handling and intensity post-processing
for VV SAR backscatter images.

Public API
──────────
• build_land_masker(shapefile_path) -> callable
• mask_land_and_clip(src, land_masker, clip_percentile, dilate_px) -> np.ndarray
• dilate_land_mask(arr, n_pixels)  (exposed for reuse)

"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely.geometry import box, mapping

from src.utils.raster_io import read_raster_data, get_nodata_from_src

MASK_VALUE = 0.0  # value committed to disk


def build_land_masker(shapefile: str) -> Callable[[rasterio.CRS, dict], list[dict]]:
    """
    Load land polygons once; return a closure that clips/reprojects
    them on demand for *any* UTM scene.

    Returns
    -------
    A function that takes a rasterio CRS and bounds tuple,
    and returns a list of GeoJSON-like dicts representing the land mask.
    """
    land_ll = gpd.read_file(shapefile)
    if land_ll.crs is None:
        land_ll = land_ll.set_crs("EPSG:4326")

    @lru_cache(maxsize=256)
    def _clip_to_scene(
        crs_wkt: str, bounds_tuple: tuple[float, float, float, float]
    ) -> tuple[dict, ...]:
        minx, miny, maxx, maxy = bounds_tuple
        footprint_ll = box(minx, miny, maxx, maxy)
        clipped = gpd.clip(
            land_ll, gpd.GeoDataFrame(geometry=[footprint_ll], crs="EPSG:4326")
        )
        if clipped.empty:
            return tuple()
        os.environ["OGR_ENABLE_PARTIAL_REPROJECTION"] = "TRUE"
        try:
            return tuple(
                transform_geom("EPSG:4326", crs_wkt, mapping(geom))
                for geom in clipped.geometry
                if geom.is_valid
            )
        except Exception:
            return tuple()
        finally:
            os.environ.pop("OGR_ENABLE_PARTIAL_REPROJECTION", None)

    def _land_masker(crs: rasterio.CRS, bounds) -> list[dict]:
        return list(_clip_to_scene(crs.to_string(), bounds))

    return _land_masker


def mask_land_and_clip(
    src: rasterio.DatasetReader,
    land_masker: Callable[
        [rasterio.CRS, tuple[float, float, float, float]], list[dict]
    ],
    *,
    clip_percentile: float | None | Tuple[float, float] = 99.9,
    dilate_px: int = 0,
) -> np.ndarray:
    """Main algorithm: (1) nodata→NaN (2) land mask (3) dilate (4) optional clip."""
    shapes_utm = land_masker(src.crs, _native_bounds_as_ll(src))

    if not shapes_utm:  # ocean-only dataset
        out = read_raster_data(src)
    else:
        masked_arr, _ = rio_mask(
            src, shapes_utm, invert=True, nodata=np.nan, filled=True, crop=False
        )
        out = masked_arr[0].astype("float32")
        out[out == get_nodata_from_src(src)] = np.nan  # ensure nodata is NaN

    if dilate_px > 0:
        out = dilate_land_mask(out, dilate_px)

    if clip_percentile is not None:
        if isinstance(clip_percentile, tuple):
            lo, hi = np.nanpercentile(out, clip_percentile)
        elif isinstance(clip_percentile, (int, float)):
            lo = np.nanpercentile(out, 0)
            hi = np.nanpercentile(out, clip_percentile)
        else:
            raise ValueError(
                f"clip_percentile must be a tuple or a number, got {clip_percentile}"
            )
        out = np.clip(out, lo, hi)

    return out


def dilate_land_mask(arr: np.ndarray, n_pixels: int = 2) -> np.ndarray:
    """Morphologically grow NaN (land) regions by `n_pixels` in all directions."""
    land = np.isnan(arr)
    footprint = np.ones((2 * n_pixels + 1,) * 2, bool)
    dilated = ndimage.binary_dilation(land, structure=footprint)
    out = arr.copy()
    out[dilated] = np.nan
    return out


def _native_bounds_as_ll(
    src: rasterio.DatasetReader,
) -> tuple[float, float, float, float]:
    """Return (left, bottom, right, top) in EPSG:4326."""
    return rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)  # type: ignore[arg-type]

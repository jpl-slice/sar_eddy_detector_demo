#!/usr/bin/env python3
"""
Mask land + (optional) high-tail clip for each selected SAR frame.

* Reads config from Hydra configuration
* Always writes a 1024-row PNG preview in data/visualisations/
* Optionally writes the masked GeoTIFF (dtype = float32/uint16/uint8)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from .transforms import MASK_VALUE, build_land_masker, mask_land_and_clip


def preprocess_frame(
    raw_tif: Path,
    out_tif: Path,
    preview_png: Path,
    land_masker,
    config,
):
    """Mask land, optional dB conversion, save masked TIFF and preview. Config can be dict or namespace."""

    # make sure all parent folders exist
    out_tif.parent.mkdir(exist_ok=True, parents=True)
    preview_png.parent.mkdir(exist_ok=True, parents=True)

    def _get(name):
        """Get config value, handling both dict-like and attribute access."""
        if hasattr(config, name):
            return getattr(config, name)
        elif isinstance(config, dict) and name in config:
            return config[name]
        else:
            raise KeyError(f"Configuration key '{name}' not found")

    clip = _get("clip_percentile")
    dilate = _get("dilate_px")
    db = _get("convert_to_db")
    save = _get("save_masked")
    dtype = _get("masked_dtype")
    compress = _get("compress")
    with rio.open(raw_tif) as src:
        masked = mask_land_and_clip(
            src, land_masker, clip_percentile=clip, dilate_px=dilate
        )
        if db:
            eps = 1e-10
            valid = np.isfinite(masked)
            masked[valid] = 10 * np.log10(masked[valid] + eps)
            masked = np.clip(masked, -35, None)
        if save:
            write_masked(src, masked, out_tif, dtype, compress=compress)
        quicklook(src, masked, preview_png)
    return out_tif, preview_png


def extract_scene_id(filename: str) -> str:
    """Extract Sentinel-1 scene ID from filename."""
    basename = Path(filename).stem
    # regex to split on known suffixes and keep only the scene ID part
    scene_id = re.split(r"_Cal|_ML|_Spk|_TC|_Orb|_processed", basename)[0]
    return scene_id


def write_masked(
    src: rasterio.DatasetReader,
    masked: np.ndarray,
    out_tif: Path,
    dtype: str = "float32",
    compress: str = "DEFLATE",
    tiled: bool = True,
):
    """Writes a masked numpy array to a GeoTIFF file."""
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=MASK_VALUE,
        compress=compress,
        tiled=tiled,
        blockxsize=512,
        blockysize=512,
    )

    if compress in {"DEFLATE", "ZSTD"}:
        profile["predictor"] = 2  # horizontal differencing
        profile["zlevel"] = 9
        profile["num_threads"] = 12

    if dtype in ("uint8", "uint16"):
        # linear min-max stretch over valid ocean pixels
        valid = np.isfinite(masked)
        data = scale_valid_masked_data(masked, dtype, valid)
        data[~valid] = MASK_VALUE  # set nans to nodata value
    else:  # float32 passthrough
        data = np.where(np.isfinite(masked), masked, MASK_VALUE).astype("float32")

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data, 1)

    return data


def scale_valid_masked_data(
    masked: np.ndarray,
    dtype: str,
    valid: np.ndarray,
    percentile: tuple[float, float] = (0, 99),
) -> np.ndarray:
    """Scale valid masked data to the range of the specified dtype.

    To brighten the image and sacrifice information at the brightest parts,
    use a lower percentile for the high end. (i.e., lower white point).
    This will stretch the narrower band of  data from low_percentile - high_percentile over
    the same uint8/uint16 range.

    Args:
        masked (np.ndarray): The masked data array with NaNs for land.
        dtype (str): The desired output data type, e.g., "uint8" or "uint16".
        valid (np.ndarray): A boolean mask indicating valid pixels in `masked`.
        percentile (tuple[float, float]): Percentiles to use for scaling (default: (0, 99)).
    Returns:
        np.ndarray: The scaled data array with the same shape as `masked`,
                    with NaNs replaced by the nodata value.
    """
    if not np.any(valid):
        return np.zeros_like(masked, dtype=dtype)

    lo, hi = np.nanpercentile(masked[valid], percentile)
    if lo == hi:
        # all valid pixels have the same value, scale to 1

        scaled = np.zeros_like(masked, dtype=dtype)
        scaled[valid] = 1
        return scaled

    scale = 255 if dtype == "uint8" else 65535
    usable_range = scale - 1
    denom = hi - lo if hi != lo else 1.0
    scaled = np.zeros_like(masked, dtype=dtype)
    scaled[valid] = (
        np.round(((masked[valid] - lo) / denom) * usable_range) + 1
    ).astype(dtype)
    return scaled


def quicklook(
    src: rasterio.DatasetReader,
    masked: np.ndarray,
    out_png: Path,
    rows: int = 2048,
):
    """Saves a lat-lon referenced grayscale preview PNG."""
    scale = rows / src.height
    cols = max(1, int(src.width * scale))

    # resample masked array using rasterio.io
    mem_profile = src.profile | {"driver": "MEM", "nodata": MASK_VALUE}
    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:
        with memfile.open(**mem_profile) as mem:
            mem.write(masked, 1)
            arr = mem.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)

    # 2) Now, figure out the lon/lat bounds for plotting
    #    Rasterio’s `src.bounds` is in src.crs. If src.crs is not geographic,
    #    we reproject those bounds into EPSG:4326.
    try:
        crs = src.crs
    except AttributeError:
        crs = None

    if crs is None:
        b = src.bounds
        left, bottom, right, top = (b.left, b.bottom, b.right, b.top)
    else:
        left, bottom, right, top = transform_bounds(crs, "EPSG:4326", *src.bounds)

    # Now (left,bottom,right,top) are in lon/lat
    # 3) Build our lon & lat arrays (in degrees) exactly as before, but now
    #    they represent true longitudes and latitudes.
    lon = np.linspace(left, right, cols)
    lat = np.linspace(top, bottom, rows)

    # create matplotlib image with lat-lon coordinates
    fig, ax = plt.subplots(figsize=(cols / 200, rows / 200), dpi=300)
    # lat = np.linspace(src.bounds.top, src.bounds.bottom, rows)  # top to bottom
    # lon = np.linspace(src.bounds.left, src.bounds.right, cols)  # left to right
    im = ax.imshow(
        np.nan_to_num(arr),
        extent=(lon[0], lon[-1], lat[0], lat[-1]),
        cmap="gray",
        vmin=np.nanmin(arr),
        vmax=np.nanmax(arr),
    )
    ax.set_aspect("equal")
    ax.set(title=Path(src.name).name, xlabel="Longitude", ylabel="Latitude")
    plt.tight_layout()
    fig.colorbar(im, ax=ax, label="SAR intensity", fraction=0.046, pad=0.04)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # simple CLI for standalone usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_tif", type=Path)
    parser.add_argument("out_tif", type=Path)
    parser.add_argument("preview_png", type=Path)
    parser.add_argument("land_shapefile", type=Path)
    parser.add_argument("--clip", type=float, default=99.9)
    parser.add_argument("--dilate", type=int, default=12)
    parser.add_argument("--db", action="store_true")
    parser.add_argument("--no-save", dest="save_masked", action="store_false")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--compress", default="DEFLATE")
    args = parser.parse_args()
    lm = build_land_masker(args.land_shapefile)
    # build a params dict matching expected keys
    params = {
        "clip_percentile": args.clip,
        "dilate_px": args.dilate,
        "convert_to_db": args.db,
        "save_masked": args.save_masked,
        "masked_dtype": args.dtype,
        "compress": args.compress,
    }
    preprocess_frame(args.raw_tif, args.out_tif, args.preview_png, lm, params)

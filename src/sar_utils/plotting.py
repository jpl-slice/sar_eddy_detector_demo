from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

# --- Core Reusable Plotting Function ---


def _create_georeferenced_plot(
    raster_array: np.ndarray,
    src: rasterio.DatasetReader,
    title: str,
    out_png: Path,
    bounding_boxes: Optional[list] = None,
    add_colorbar: bool = False,
):
    """
    Creates and saves a georeferenced plot with lat-lon axes.

    This function is the single source of truth for plotting. It handles
    coordinate reprojection, plotting the raster, optionally adding bounding
    boxes and a colorbar, and saving the figure.
    """
    rows, cols = raster_array.shape
    fig, ax = plt.subplots(figsize=(cols / 200, rows / 200), dpi=300)

    # Figure out the lon/lat bounds for plotting
    #    Rasterio’s `src.bounds` is in src.crs. If src.crs is not geographic,
    #    we reproject those bounds into EPSG:4326.
    crs = getattr(src, "crs", None)

    if crs is None:
        b = src.bounds
        left, bottom, right, top = (b.left, b.bottom, b.right, b.top)
    else:
        left, bottom, right, top = transform_bounds(crs, "EPSG:4326", *src.bounds)
    im = ax.imshow(
        np.nan_to_num(raster_array),
        extent=(left, right, bottom, top),  # Use reprojected bounds
        cmap="gray",
        vmin=np.nanmin(raster_array),
        vmax=np.nanpercentile(raster_array, 98),
    )
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="Longitude", ylabel="Latitude")

    if bounding_boxes:
        # Bounding boxes must also be reprojected to be displayed correctly
        for xmin, ymin, xmax, ymax in bounding_boxes:
            if crs is not None:
                xmin, ymin, xmax, ymax = transform_bounds(
                    crs, "EPSG:4326", xmin, ymin, xmax, ymax
                )
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

    if add_colorbar:
        fig.colorbar(im, ax=ax, label="SAR intensity", fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

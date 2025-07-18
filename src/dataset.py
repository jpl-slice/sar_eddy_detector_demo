import contextlib
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio import windows
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm

from src.utils.file_preprocess_checker import check_file_is_preprocessed
from src.utils.raster_io import read_raster_data


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


def _process_single_image_wrapper(
    tif_path: str,
    win_size: int,
    stride: int,
    nodata_threshold: float,
) -> Tuple[List[rasterio.windows.Window], List[str]]:
    """
    Stand-alone so it can run in a different process.
    Re-creates only the cheap state it needs; avoids pickling the whole Dataset.
    """
    # Check if file is preprocessed before processing
    check_file_is_preprocessed(tif_path, require_preprocessing=True)
    dataset = object.__new__(SARTileDataset)  # empty shell
    dataset.win_size, dataset.stride = win_size, stride
    dataset.nodata_threshold = nodata_threshold

    # call the existing code path – returns Booleans etc. as usual
    dataset._initialize_raster_properties(Path(tif_path))
    tile_windows = dataset._compute_initial_windows_optimized_numpy()
    paths = [tif_path] * len(tile_windows)
    return tile_windows, paths


class SARTileDataset(Dataset):
    """
    A sliding-window detection dataset over preprocessed SAR GeoTIFF(s),
    returning a dictionary with the following keys:
    - "image": HxWx3 numpy array of the tile.
    - "filename": Path to the source GeoTIFF file.
    - "bbox": Bounding box in pixel coordinates [px_min, py_min, px_max, py_max].
    - "bbox_geo": Bounding box in geographic coordinates [lon_min, lat_min, lon_max, lat_max].
    - "crs": Coordinate Reference System as a string.
    - "transform": Affine transform for the window.

    NOTE: This version assumes that all GeoTIFFs in `dataset_config["geotiff_dir"]`
    have already been preprocessed (e.g., land masked, scaled) externally.

    Image File Handling:
    -----------------------------------
    The dataset can be initialized in a few ways depending on how image paths
    are specified and where the image files are located:

    1. `dataset_config['geotiff_dir']` is a directory path (e.g., "/path/to/images/"):
       - The dataset will look for image (.tif and .tiff) files within this specified directory.

    2. `dataset_config['geotiff_dir']` is a path to a single image file (e.g., "/path/to/images/image1.tif"):
       - The dataset will process only this single image file.

    Attributes:
        win_size (int): Size of the sliding window (tile height and width).
        stride (int): Stride of the sliding window.
        transform (Optional[Compose]): Torchvision transforms to apply to each tile.
        img_size (int): Target image size after potential resizing (used for mosaic border,
                        though mosaic is currently disabled).
        _windows (List[windows.Window]): List of rasterio window objects for valid tiles.
        _window_paths (List[str]): List of source GeoTIFF paths for each window.
        img_files (List[str]): Generated unique names for each tile/window.
        n (int): Total number of valid tiles.
        indices (List[int]): List of indices for data shuffling/access.
    """

    def __init__(self, dataset_config: Dict, transform: Optional[Callable] = None):
        """
        Args:
            dataset_config: A dictionary containing all dataset parameters.
                - geotiff_dir: Directory containing raw GeoTIFF files.
                - land_shapefile: Path to land polygon shapefile.
                - window_size: Size of tiles to extract (square).
                - stride_factor: Stride as a fraction of window size.
                - land_threshold: Maximum fraction of land pixels allowed.
                - nodata_threshold: Maximum fraction of no-data pixels allowed.
                - var_threshold: Minimum variance required for valid tiles.
            transform: PyTorch transforms to apply to tiles (composed outside).
        """
        self.geotiff_dir = dataset_config["geotiff_dir"]
        self.preprocessed_dir = (
            self.geotiff_dir
        )  # keep this attribute for compatibility
        self.win_size = dataset_config.get("window_size", 448)
        self.stride = int(dataset_config.get("stride_factor", 0.5) * self.win_size)
        self.transform = transform
        self.img_size = dataset_config.get("img_size", self.win_size)
        self.nodata_threshold = dataset_config.get("nodata_threshold", 0.75)

        # --- these start as Python lists, converted to NumPy at the end ----
        self._windows = []
        self.shapes: List = []
        self._window_paths: List[str] = []

        image_input_path = self.geotiff_dir
        if image_input_path is None:
            raise ValueError("No images found")

        p = Path(image_input_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Provided image_input_path does not exist: {image_input_path}"
            )
        if p.is_dir():
            tif_paths = sorted(
                [
                    str(path)
                    for ext in ("*.tif", "*.tiff", "*.jp2")
                    for path in p.glob(ext)
                ]
            )
            # Decide how many workers you can afford
            max_workers = min(os.cpu_count() or 1, 48)  # ~250 GB / 8 GB ≈ 31
            # Submit jobs
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        _process_single_image_wrapper,
                        tif_path,
                        self.win_size,
                        self.stride,
                        self.nodata_threshold,
                    )
                    for tif_path in tif_paths
                ]

                for f in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing images (parallel)",
                ):
                    wins, paths = f.result()
                    if wins:
                        self._windows.extend(wins)
                        self._window_paths.extend(paths)
        else:  # Single file mode
            check_file_is_preprocessed(str(p), require_preprocessing=True)
            self._process_single_image(p)

        if not self._windows:
            raise ValueError("No valid windows were generated.")

        # windows → (N,4) int32  [col_off,row_off,w,h]
        self._windows = np.asarray(
            [[w.col_off, w.row_off, w.width, w.height] for w in self._windows],
            dtype=np.int32,
        )

        # window paths → fixed-width unicode array
        max_len = max(len(p) for p in self._window_paths)
        self._window_paths = np.asarray(self._window_paths, dtype=f"U{max_len}")
        # img_files → unicode array
        self.img_files = np.asarray(
            [f"{Path(fp).stem}_win{i}" for i, fp in enumerate(self._window_paths)],
            dtype=f"U{max_len+8}",
        )
        self.n = self._windows.shape[0]
        self.indices = np.arange(self.n, dtype=np.int32)

        # per‐worker cache {worker_id: {path: (DatasetReader, nodata_val)}}
        self._srcs: Dict[int, Dict[str, Tuple[rasterio.DatasetReader, float]]] = {}

        print(
            f"""
        Dataset Summary:
        - Total .tif files processed: {len(set(self._window_paths))}
        - Total valid windows: {self.n}
        - Window size: {self.win_size}x{self.win_size}
        - Stride: {self.stride}
        """
        )

    def __len__(self) -> int:
        return self.n

    def _process_single_image(self, tif_path: Path) -> bool:
        """Process one GeoTIFF and extend class-level lists in-place. Returns True if successful."""
        try:
            self._initialize_raster_properties(tif_path)
            windows = self._compute_initial_windows_optimized_numpy()
            if windows:
                self._windows.extend(windows)
                self._window_paths.extend([str(tif_path)] * len(windows))
            return True
        except Exception as e:
            print(f"Error processing single image {tif_path}: {e}")
            return False
        finally:
            if hasattr(self, "src") and self.src is not None:
                with contextlib.suppress(Exception):
                    self.src.close()
                self.src = None

    def _initialize_raster_properties(self, geotiff_path: Path):
        self.src = rasterio.open(str(geotiff_path))
        self.nodata_val = get_nodata_from_src(self.src)

    def _compute_initial_windows(self) -> List[windows.Window]:
        h, w = self.src.height, self.src.width
        initial_windows_list: List[windows.Window] = []
        if h < self.win_size or w < self.win_size:
            return []

        for y_offset in range(0, h - self.win_size + 1, self.stride):
            for x_offset in range(0, w - self.win_size + 1, self.stride):
                win = windows.Window(x_offset, y_offset, self.win_size, self.win_size)
                data = self.src.read(
                    1, window=win, boundless=True, fill_value=self.nodata_val
                )
                if (data == self.nodata_val).mean() < self.nodata_threshold:
                    initial_windows_list.append(win)
        return initial_windows_list

    # @timeit
    def _compute_initial_windows_optimized(self) -> List[windows.Window]:
        """
        Computes initial valid windows by loading the entire raster into memory
        and using fully vectorized operations.

        WARNING: This will fail with a MemoryError on very large GeoTIFFs.
        Use only if you are certain all your images can fit in RAM.
        """
        h, w = self.src.height, self.src.width
        if h < self.win_size or w < self.win_size:
            return []

        # --- 1. Load the ENTIRE image into memory ---
        # DANGER: This is the step that can cause a MemoryError.
        # print(f"Loading entire {w}x{h} raster into memory...")
        full_image_data = self.src.read(1)

        # --- 2. Create the nodata mask for the full image ---
        nodata_mask = (full_image_data == self.nodata_val).astype(np.float32)
        del full_image_data  # Free up memory as soon as possible

        # --- 3. Use uniform_filter to get nodata fraction for all possible windows ---
        # This remains the most efficient way to calculate the metric.
        nodata_fraction_map = uniform_filter(
            nodata_mask, size=self.win_size, mode="constant", cval=1.0
        )

        # --- 4. Generate a grid of all potential window start coordinates ---
        # These are the top-left (y, x) coordinates for every window, respecting the stride.
        y_starts = np.arange(0, h - self.win_size + 1, self.stride)
        x_starts = np.arange(0, w - self.win_size + 1, self.stride)
        yy, xx = np.meshgrid(y_starts, x_starts, indexing="ij")

        # --- 5. Check the condition for all windows at once (fully vectorized) ---
        # Get the nodata fraction at the bottom-right corner of each window in our grid.
        window_end_y = yy + self.win_size - 1
        window_end_x = xx + self.win_size - 1

        # Get the fractions for all windows in our stride grid in one operation.
        all_fractions = nodata_fraction_map[window_end_y, window_end_x]

        # Create a boolean mask of valid windows.
        valid_mask = all_fractions < self.nodata_threshold

        # --- 6. Select the coordinates of the valid windows ---
        valid_y_starts = yy[valid_mask]
        valid_x_starts = xx[valid_mask]

        # --- 7. Create the final list of rasterio.windows.Window objects ---
        final_windows_list = [
            windows.Window(int(x), int(y), self.win_size, self.win_size)
            for y, x in zip(valid_y_starts, valid_x_starts)
        ]

        return final_windows_list

    def _compute_initial_windows_optimized_numpy(self) -> List[windows.Window]:
        """
        Vectorized exact nodata filtering via summed‐area table (integral image).
        Returns the same List[windows.Window] as _compute_initial_windows().
        """
        h, w = self.src.height, self.src.width
        win, stride = self.win_size, self.stride

        # no possible windows
        if h < win or w < win:
            return []

        # 1. Read full band and build a binary nodata mask
        data = read_raster_data(self.src)
        mask = ((data == self.nodata_val) | np.isnan(data)).astype(np.uint32)

        # 2. Build summed‐area table with a zero‐pad row/col at top+left
        S = mask.cumsum(axis=0).cumsum(axis=1)
        S = np.pad(S, ((1, 0), (1, 0)), mode="constant", constant_values=0)

        # 3. Generate all window‐start coordinates
        y0s = np.arange(0, h - win + 1, stride)
        x0s = np.arange(0, w - win + 1, stride)
        yy, xx = np.meshgrid(y0s, x0s, indexing="ij")

        # 4. Compute bottom‐right corners
        y1 = yy + win
        x1 = xx + win

        # 5. Exact nodata‐pixel counts via inclusion‐exclusion
        #    sums = S[y0, x0] + S[y1, x1] - S[y1, x0] - S[y0, x1]
        sums = S[yy, xx] + S[y1, x1] - S[y1, xx] - S[yy, x1]
        # 6. Convert to fractions and threshold
        fractions = sums.astype(np.float32) / (win * win)
        valid = fractions < self.nodata_threshold

        # 7. Build windows list
        valid_windows = [
            windows.Window(int(x), int(y), win, win)
            for y, x in zip(yy[valid], xx[valid])
        ]
        return valid_windows

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns a dictionary containing the data for a single tile."""
        idx = self.indices[index]

        # Convert to tensor but let transforms handle normalization
        # bring to [0,1] if needed; TODO: figure out what to do if we pass in raw dB values (e.g.., -20, -15, etc.)

        col_off, row_off, w, h = self._windows[idx]
        win = windows.Window(col_off, row_off, w, h)
        tif_path = self._window_paths[idx]
        tif_path_str = str(tif_path)

        # Read the data for the specified window
        src, nodata = self._get_src(tif_path)
        arr = read_raster_data(src, window=win)
        img_hwc = np.stack([arr] * 3, axis=-1)

        if np.all(img_hwc == 0):
            raise ValueError(
                f"Processed image from {tif_path_str} is all zeros after scaling. Check nodata handling or input data."
            )
        window_transform = src.window_transform(win)
        window_transform = [
            getattr(window_transform, x) for x in ["a", "b", "c", "d", "e", "f"]
        ]  # Affine transform

        # Apply transforms if provided (this is where normalization should happen)
        if self.transform is not None:
            img_tensor = self.transform(arr)
        else:
            img_tensor = torch.from_numpy(arr).float()
        img_tensor = torch.nan_to_num(img_tensor)  # set nan to 0

        # Get bounding boxes in both pixel and geographic coordinates
        bbox_pixel, bbox_geo = self._get_bounding_boxes(src, col_off, row_off, w, h)
        return {
            "image": img_tensor,  # H×W×3 numpy array
            "filename": tif_path_str,
            "bbox": bbox_pixel,  # Bounding box in pixel format
            "bbox_geo": bbox_geo,  # Bounding box in geographic format
            "crs": src.crs.to_string(),  # Coordinate Reference System as a string
            "transform": torch.tensor(window_transform),
        }

    def _get_src(self, path: str):
        # Manages opening files on a per-worker basis
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # Check if this worker has a cache for this file path
        if worker_id not in self._srcs:
            self._srcs[worker_id] = {}

        if path not in self._srcs[worker_id]:
            # This worker is opening this file for the first time
            src = rasterio.open(path)
            nodata = get_nodata_from_src(src)
            self._srcs[worker_id][path] = (src, nodata)
        return self._srcs[worker_id][path]

    def _get_bounding_boxes(
        self, src: rasterio.DatasetReader, col_off: int, row_off: int, w: int, h: int
    ) -> Tuple[List[int], List[float]]:
        """For a given tile, extracts bounding box information
        in both pixel and geographic (lat/lon) coordinates.

        Args:
            src (rasterio.DatasetReader): _rasterio DatasetReader object for the source GeoTIFF.
            col_off (int): Column offset of the tile in the source image.
            row_off (int): Row offset of the tile in the source image.
            w (int): Width of the tile in pixels.
            h (int): Height of the tile in pixels.

        Returns:
            Tuple[List[float], List[float]]: Bounding box coordinates in pixel and geographic formats.
        """
        # 1. Define pixel coordinates for the tile's bounding box relative to the full image.
        # The top-left corner of the window is (col_off, row_off).
        # The bottom-right corner is (col_off + width, row_off + height).
        px_min, py_min = col_off, row_off
        px_max, py_max = col_off + w, row_off + h

        # 2. Use rasterio.transform.xy to get geographic coordinates for the corners.
        # The `xy` method takes row, col, so the order is (py, px).
        # We get (lon, lat) for the top-left and bottom-right corners.
        lon_min, lat_max = src.transform * (px_min, py_min)
        lon_max, lat_min = src.transform * (px_max, py_max)

        # 3. Assemble the bounding boxes.
        # Pixel bbox is relative to the full source image.
        # Geographic bbox is [lon_min, lat_min, lon_max, lat_max].
        pixel_bbox = torch.Tensor([px_min, py_min, px_max, py_max])
        geo_bbox = torch.Tensor([lon_min, lat_min, lon_max, lat_max])

        return pixel_bbox, geo_bbox

    def __del__(self):
        # Optional but good practice: ensure all file handles are closed when the object is destroyed.
        if hasattr(self, "_srcs"):
            for worker_files in self._srcs.values():
                for src, _ in worker_files.values():
                    try:
                        src.close()
                    except:
                        pass

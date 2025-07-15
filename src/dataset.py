import concurrent.futures
import glob
import os
import shutil
import tempfile
from typing import Callable, Dict, Optional

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio import features, windows
from torch.utils.data import Dataset


class SARTileDataset(Dataset):
    """
    Dataset for extracting valid ocean tiles from SAR GeoTIFF images.

    Supports:
      1. Preprocessing raw GeoTIFFs: Optionally apply the land mask once and save files with land pixels set to nodata.
         Use the `preprocessed_dir` argument to indicate where to save (or read from) these files.
      2. Persistent file handle caching: Opens files once per worker and reuses the handle across __getitem__ calls.

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
                - preprocessed_dir: Optional directory for preprocessed (land-masked) GeoTIFFs.
                - force_preprocess: If True, reprocess raw files even if preprocessed files exist.
            transform: PyTorch transforms to apply to tiles (composed outside).
        """
        # Unpack config dictionary
        self.geotiff_dir = dataset_config["geotiff_dir"]
        self.land_shapefile = dataset_config["land_shapefile"]
        self.window_size = dataset_config.get("window_size", 448)
        self.stride = int(dataset_config.get("stride_factor", 0.5) * self.window_size)
        self.land_threshold = dataset_config.get("land_threshold", 0.8)
        self.nodata_threshold = dataset_config.get("nodata_threshold", 0.9)
        self.var_threshold = dataset_config.get("var_threshold", 1e-5)
        self.preprocessed_dir = dataset_config.get("preprocessed_dir")
        force_preprocess = dataset_config.get("force_preprocess", False)

        # Dictionary to cache persistent file handles (each worker gets its own instance)
        self._file_handles = {}
        # Cache for any per-file land mask (if needed for raw files)
        self.land_mask_file_cache = {}

        # Load land polygons once
        print(f"Loading land polygons from {self.land_shapefile}")
        self.land_gdf = gpd.read_file(self.land_shapefile)
        if self.land_gdf.crs is None:
            self.land_gdf = self.land_gdf.set_crs("EPSG:4326")

        if self.preprocessed_dir is None:  # use temp dir
            self.preprocessed_dir = tempfile.mkdtemp()
            print(
                f"Using temporary directory for preprocessed files: {self.preprocessed_dir}"
            )

        # If preprocessed_dir is provided, ensure it exists and process files if needed
        if self.preprocessed_dir is not None:
            os.makedirs(self.preprocessed_dir, exist_ok=True)
            raw_files = sorted(glob.glob(os.path.join(self.geotiff_dir, "*.tif")))
            jp2_files = sorted(glob.glob(os.path.join(self.geotiff_dir, "*.jp2")))
            raw_files = raw_files + jp2_files
            self.files = []

            # Create preprocessing tasks
            preprocess_tasks = []
            for raw_file in raw_files:
                base = os.path.basename(raw_file)
                pre_file = os.path.join(self.preprocessed_dir, base)
                if not os.path.exists(pre_file) or force_preprocess:
                    preprocess_tasks.append((raw_file, pre_file, base))
                else:
                    print(f"Using preprocessed file: {base}")
                self.files.append(pre_file)

            # Process files in parallel if any need preprocessing
            if preprocess_tasks:
                cpus = os.cpu_count()
                workers = max(1, min((cpus - 1 if cpus else 1), len(preprocess_tasks)))
                print(
                    f"Preprocessing {len(preprocess_tasks)} files using {workers} workers"
                )

                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exc:
                    # Submit all preprocessing jobs
                    futures = {}
                    for raw_file, pre_file, basename in preprocess_tasks:
                        future = exc.submit(
                            self._apply_land_mask_and_scaling, raw_file, pre_file
                        )
                        futures[future] = basename

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        base = futures[future]
                        try:
                            future.result()
                            print(f"Finished preprocessing: {base}")
                        except Exception as e:
                            print(f"Error preprocessing {base}: {str(e)}")
        else:
            self.files = sorted(glob.glob(os.path.join(self.geotiff_dir, "*.tif")))
            if not self.files:
                raise ValueError(f"No GeoTIFF files found in {self.geotiff_dir}")

        print(
            f"Found {len(self.files)} GeoTIFF files (preprocessed: {self.preprocessed_dir is not None})"
        )
        print("Building tile index...")
        self.tile_index = self._build_tile_index_in_parallel()
        self.transform = transform

    def _apply_land_mask_and_scaling(self, raw_file: str, pre_file: str):
        """
        Preprocess a raw GeoTIFF file by applying the land mask (setting land pixels to nodata)
        and save the resulting file to pre_file.
        """
        with rasterio.open(raw_file) as src:
            if src.driver == "JP2OpenJPEG" or raw_file.endswith(".jp2"):
                # copy over jp2 and any metadata xmls to pre_file since these already have the land mask applied``
                shutil.copy(raw_file, pre_file)
                xml_file = f"{raw_file}.aux.xml"
                if os.path.exists(xml_file):
                    shutil.copy(xml_file, f"{pre_file}.aux.xml")
                return
            file_crs = src.crs
            file_transform = src.transform
            file_height = src.height
            file_width = src.width
            nodata_value = src.nodata if src.nodata is not None else -9999

            # Reproject land polygons to the file's CRS
            land_gdf_file_crs = self.land_gdf.to_crs(file_crs)

            # Rasterize the land polygons
            shapes = ((geom, 1) for geom in land_gdf_file_crs.geometry)
            land_mask = features.rasterize(
                shapes=shapes,
                out_shape=(file_height, file_width),
                fill=0,
                transform=file_transform,
                all_touched=True,
                dtype=np.uint8,
            )

            # Read the entire data and set land pixels to nodata
            data = src.read(1).astype(np.float32)
            data[land_mask == 1] = nodata_value

            # Update metadata to reflect nodata value
            meta = src.meta.copy()
            meta.update({"nodata": nodata_value, "dtype": "float32"})

            with rasterio.open(pre_file, "w", **meta) as dst:
                dst.write(data, 1)

    def _build_tile_index_in_parallel(self):
        """
        Build an index of valid tiles across all files using multiprocessing.
        If files are preprocessed, land masking is already done so __getitem__ can be simplified.
        """
        all_tile_indices = []
        num_cpus = os.cpu_count()
        max_workers = max(1, min((num_cpus - 1 if num_cpus else 1), len(self.files)))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(self._find_valid_tile_indices_from_file, file_path, idx)
                for idx, file_path in enumerate(self.files)
            ]
            for future in concurrent.futures.as_completed(futures):
                file_indices, _, _ = future.result()
                if file_indices:
                    all_tile_indices.extend(file_indices)
        print(
            f"Total: {len(all_tile_indices)} valid tiles across {len(self.files)} files"
        )
        return all_tile_indices

    def _find_valid_tile_indices_from_file(self, file_path, file_idx):
        """
        Process a single file to build tile indices.
        When using preprocessed files, land pixels have already been masked.
        """
        # print(
        #     f"Processing file {file_idx+1}/{len(self.files)}: {os.path.basename(file_path)}"
        # )
        file_tile_index = []
        try:
            with rasterio.open(file_path) as src:
                file_height, file_width = src.height, src.width
                steps_x = max(1, (file_width - self.window_size) // self.stride + 1)
                steps_y = max(1, (file_height - self.window_size) // self.stride + 1)
                valid_tiles = 0
                for i_step in range(steps_y):
                    for j_step in range(steps_x):
                        row_start = i_step * self.stride
                        col_start = j_step * self.stride
                        row_end = min(row_start + self.window_size, file_height)
                        col_end = min(col_start + self.window_size, file_width)
                        if (row_end - row_start) < self.window_size or (
                            col_end - col_start
                        ) < self.window_size:
                            continue
                        current_window = ((row_start, row_end), (col_start, col_end))
                        data = src.read(1, window=current_window)
                        total_pixels = self.window_size * self.window_size
                        if src.nodata is not None:
                            nodata_pixels = np.sum(data == src.nodata)
                        else:
                            nodata_pixels = 0
                        nodata_percentage = nodata_pixels / total_pixels
                        if (
                            nodata_percentage < self.nodata_threshold
                            and np.var(data) > self.var_threshold
                        ):
                            file_tile_index.append(
                                {
                                    "file_path": file_path,
                                    "window": current_window,
                                }
                            )
                            valid_tiles += 1
                print(
                    f"  Found {valid_tiles} valid tiles in {os.path.basename(file_path)}"
                )

                return file_tile_index, file_path, None
        except Exception as e:
            print(f"Error finding valid tile indices for {file_path}: {str(e)}")
            return [], None, None

    def _get_file_handle(self, file_path):
        """
        Retrieve a persistent file handle for the given file.
        """
        if file_path not in self._file_handles:
            self._file_handles[file_path] = rasterio.open(file_path)
        return self._file_handles[file_path]

    def __len__(self):
        return len(self.tile_index)

    def expand_from_jp2(self, src, window=None):
        comp_data_uint16 = src.read(1, window=window).astype(np.float32)
        tags = src.tags()

        # Retrieve min/max from tags or use fallback values
        ocean_min = float(tags.get("min_val", 0))
        ocean_max = float(tags.get("max_val", 1))
        rng = ocean_max - ocean_min if ocean_max != ocean_min else 1.0

        # 0 => nodata
        comp_valid = comp_data_uint16 > 0
        comp_data_uint16[~comp_valid] = np.nan

        comp_data = (comp_data_uint16 / 65535.0) * rng + ocean_min
        return comp_data

    def __getitem__(self, idx):
        if idx >= len(self.tile_index):
            raise IndexError(f"Index out of range ({idx} >= {len(self.tile_index)})")
        tile_info = self.tile_index[idx]
        file_path = tile_info["file_path"]
        window = tile_info["window"]

        # Use the persistent file handle
        fh = self._get_file_handle(file_path)
        if file_path.endswith("jp2") or fh.driver == "JP2OpenJPEG":
            data = self.expand_from_jp2(fh, window=window)
        else:
            data = fh.read(1, window=window).astype(np.float32)
        # Convert nodata values to np.nan
        nodata_val = fh.nodata if fh.nodata is not None else -9999
        data[data == nodata_val] = np.nan

        if self.transform:
            tensor_data = self.transform(data)
        else:
            tensor_data = torch.from_numpy(data).unsqueeze(0)

        # --- Metadata Extraction ---
        window_transform = windows.transform(window, fh.transform)
        filename = os.path.basename(file_path)
        left, top = window_transform * (0, 0)
        right, bottom = window_transform * (self.window_size, self.window_size)
        bbox = torch.tensor((left, bottom, right, top))
        window_transform = [
            getattr(window_transform, x) for x in ["a", "b", "c", "d", "e", "f"]
        ]  # Affine transform
        return {
            "image": tensor_data,
            "filename": filename,
            "bbox": bbox,
            # "bbox_latlon": bbox_latlon,
            "crs": fh.crs.to_string(),
            "transform": torch.tensor(window_transform),
        }

    def close(self):
        """
        Close all persistent file handles.
        """
        if hasattr(self, "_file_handles"):
            for fh in self._file_handles.values():
                fh.close()
            self._file_handles = {}

    def __del__(self):
        self.close()

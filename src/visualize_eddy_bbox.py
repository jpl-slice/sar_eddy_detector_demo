import abc
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.sar_utils.plotting import _create_georeferenced_plot
from src.transforms import ClipNormalizeCastToUint8
from src.utils.bbox import parse_bbox
from src.utils.raster_io import read_raster_data

if TYPE_CHECKING:
    from src.eddy_detector.base_detector import BaseEddyDetector


class EddyVisualizer:
    def __init__(self, detector: "BaseEddyDetector"):
        self.detector = detector
        self.config = detector.config
        self.class_name = self.__class__.__name__
        self.normalizer = ClipNormalizeCastToUint8()

    def create_scene_previews_with_bbox(self, confidence_threshold=0.99, merged=False):
        df = self._load_and_prepare_detections_df(merged, confidence_threshold)
        if df is None:
            return

        output_prefix = f"full_scene_previews{'_merged_bbox' if merged else ''}"
        output_subdir = f"{output_prefix}_{confidence_threshold}"
        output_dir = Path(self.config.output_dir) / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.detector.dataset or not getattr(
            self.detector.dataset, "geotiff_dir", None
        ):
            print(
                f"[{self.class_name}] Dataset.geotiff_dir is required for preview generation."
            )
            return

        source_image_folder = Path(self.detector.dataset.geotiff_dir)
        processed_files_count = 0

        for filename_base in df["filename"].unique():
            file_path = source_image_folder / str(filename_base)
            if not file_path.exists():
                print(
                    f"[{self.class_name}] Warning: Source image not found, skipping preview for {filename_base}"
                )
                continue

            file_df = df[df["filename"] == filename_base]
            boxes_for_file = file_df["bbox_parsed"].tolist()
            confidences = file_df["confidence"].astype(float).tolist()
            try:
                output_image_path = output_dir / f"{file_path.stem}_preview.png"
                self._create_preview_with_boxes(
                    str(file_path),
                    boxes_for_file,
                    str(output_image_path),
                    confidences=confidences,
                    confidence_threshold=confidence_threshold,
                    scale_factor=0.1,
                )
                processed_files_count += 1
            except FileNotFoundError:
                print(
                    f"[{self.class_name}] Error generating preview: Source file not found at {file_path}"
                )
            except Exception as e:
                print(
                    f"[{self.class_name}] Error generating preview for {filename_base}: {e}"
                )
                traceback.print_exc()

        print(
            f"[{self.class_name}] Generated {processed_files_count} preview images in {output_dir}"
        )

    def save_positive_detection_tiles(
        self,
        confidence_threshold=0.9,
        merged=True,
        patch_size=128,
        bbox_buffer_pixels=32,
        normalize_window=True,
        split_scenes_into_folders=True,
    ) -> None:
        df = self._load_and_prepare_detections_df(merged, confidence_threshold)
        if df is None:
            return

        parent_dir = (
            self.detector._base_csv_path.parent
            if self.detector._base_csv_path
            else Path(self.config.output_dir)
        )
        main_output_dir = (
            parent_dir / f"positive_detection_tiles_{confidence_threshold}"
        )
        main_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.detector.dataset or not getattr(
            self.detector.dataset, "geotiff_dir", None
        ):
            print(
                f"[{self.class_name}] Dataset.geotiff_dir is required for saving individual previews."
            )
            return

        source_image_folder = Path(self.detector.dataset.geotiff_dir)
        grouped_files = df.groupby("filename")
        print(
            f"Creating individual previews for {len(df)} detections across {len(grouped_files)} files in {main_output_dir}"
        )

        for filename, group in grouped_files:
            if split_scenes_into_folders:
                file_output_dir = main_output_dir / Path(str(filename)).stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                file_output_dir = main_output_dir
            file_path = source_image_folder / str(filename)
            if not file_path.exists():
                print(
                    f"[{self.class_name}] Warning: Source image not found, skipping tile saving for {filename}"
                )
                continue

            try:
                with rasterio.open(file_path) as src:
                    for i, detection in group.iterrows():
                        bbox = detection["bbox_parsed"]
                        confidence = detection["confidence"]
                        output_path = (
                            file_output_dir
                            / f"{Path(str(filename)).stem}_detection_{i:03d}_conf_{confidence:.3f}.png"
                        )
                        try:
                            window = self._calculate_preview_window(
                                src, bbox, bbox_buffer_pixels, patch_size
                            )
                            if window is None:
                                continue
                            window_data = self._read_and_normalize_window_data(
                                src, window, normalize_window
                            )
                            if window_data is None:
                                continue
                            self._save_png_crop(window_data, output_path)
                        except Exception as inner_e:
                            print(
                                f"  Error processing detection index {i} for {filename}: {inner_e}"
                            )
            except Exception as e:
                print(
                    f"[{self.class_name}] Error processing file {filename} for individual previews: {e}"
                )
                traceback.print_exc()
        print(
            f"[{self.class_name}] All individual previews attempt finished. Check logs for errors. Output dir: {main_output_dir}"
        )

    def _load_and_prepare_detections_df(
        self, merged: bool, confidence_threshold: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        bbox_col_name = "bbox"
        csv_path = (
            self.detector._merged_csv_path if merged else self.detector._base_csv_path
        )
        source_desc = "merged" if merged else "individual"
        print(
            f"[{self.class_name}] Loading {source_desc} detections from {csv_path}..."
        )
        try:
            df = pd.read_csv(csv_path)  # type: ignore
        except (FileNotFoundError, TypeError):
            print(f"[{self.class_name}] File not found: {csv_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"[{self.class_name}] No data in CSV file: {csv_path}")
            return None
        if df.empty:
            print(
                f"[{self.class_name}] No detections found in {csv_path}. Skipping preview generation."
            )
            return None
        if bbox_col_name not in df.columns:
            print(
                f"[{self.class_name}] Error: Required column '{bbox_col_name}' not found in {csv_path}"
            )
            return None
        try:
            df["bbox_parsed"] = df[bbox_col_name].apply(parse_bbox)
        except Exception as e:
            print(f"[{self.class_name}] Error parsing bbox column in {csv_path}: {e}")
            return None
        if confidence_threshold is not None:
            original_count = len(df)
            df = df[df["confidence"] >= confidence_threshold].copy()
            filtered_count = len(df)
            print(
                f"[{self.class_name}] Filtered {original_count - filtered_count} detections below confidence threshold {confidence_threshold}"
            )
            if df.empty:
                print(f"[{self.class_name}] No detections remaining after filtering.")
        return df

    def _calculate_preview_window(
        self,
        src: rasterio.io.DatasetReader,
        bbox: Tuple[float, float, float, float],
        buffer: int,
        patch_size: int,
    ) -> Optional[Window]:
        """
        Given a WGS84 bbox (lon_min, lat_min, lon_max, lat_max),
        compute a full-resolution pixel Window in the dataset's CRS using
        transform_bounds + from_bounds, then:
        - add a pixel buffer,
        - clamp with windows.intersection,
        - enforce a minimum patch size within dataset bounds.

        Returns None if the resulting window is empty.
        """
        try:
            lon_min, lat_min, lon_max, lat_max = bbox
            # Normalize bbox ordering
            left_ll = min(lon_min, lon_max)
            right_ll = max(lon_min, lon_max)
            bottom_ll = min(lat_min, lat_max)
            top_ll = max(lat_min, lat_max)

            # Always transform bounds; no-ops if CRSes are equal.
            dst_crs = src.crs if src.crs else "EPSG:4326"
            left, bottom, right, top = rasterio.warp.transform_bounds(
                "EPSG:4326", dst_crs, left_ll, bottom_ll, right_ll, top_ll
            )

            # Convert CRS-aligned bounds to a pixel window and round to pixel grid
            win = (
                rasterio.windows.from_bounds(
                    left, bottom, right, top, transform=src.transform
                )
                .round_offsets()
                .round_lengths()
            )

            # Apply pixel buffer
            buffered = Window(
                max(0, int(win.col_off) - buffer),
                max(0, int(win.row_off) - buffer),
                int(win.width) + 2 * buffer,
                int(win.height) + 2 * buffer,
            )

            # Clamp to dataset extent using intersection
            full = Window(0, 0, src.width, src.height)
            try:
                clamped = rasterio.windows.intersection(buffered, full)
            except Exception:
                return None

            # fmt: off
            row_start, row_end = self._ensure_min_patch_size(
                int(clamped.row_off), int(clamped.row_off + clamped.height), patch_size, src.height
            )
            col_start, col_end = self._ensure_min_patch_size(
                int(clamped.col_off), int(clamped.col_off + clamped.width), patch_size, src.width
            )
            # fmt: on

            # Check for valid window
            if row_end <= row_start or col_end <= col_start:
                return None

            return Window.from_slices((row_start, row_end), (col_start, col_end))
        except Exception as e:
            print(f"[{self.class_name}] Error calculating preview window: {e}")
            return None

    def _ensure_min_patch_size(self, start, end, patch_size, max_size):
        if end - start >= patch_size:
            return start, end
        extra = patch_size - (end - start)
        start = max(0, start - extra // 2)
        end = min(max_size, start + patch_size)
        start = max(0, end - patch_size)
        return start, end

    def _read_and_normalize_window_data(
        self, src: rasterio.DatasetReader, window: Window, normalize: bool
    ) -> Optional[np.ndarray]:
        try:
            data = read_raster_data(src, window=window)
            if normalize:
                # Reuse the same normalization logic as the inference dataset
                data = self.normalizer(data)

            return data
        except Exception as e:
            print(f"[{self.class_name}] Error reading window data: {e}")
            return None

    def _save_png_crop(self, data: np.ndarray, out_path: Path) -> None:
        # Accepts either (H, W) uint8 grayscale, or (C, H, W) / (H, W, C)
        arr = data
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # (C, H, W) -> (H, W, C)
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            # Keep this simple: if normalize was skipped, fallback to simple 0–255 scaling
            arr = self.normalizer(arr)

        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            mode = "RGB" if arr.shape[2] == 3 else "RGBA"
            img = Image.fromarray(arr, mode=mode)
        else:
            # Fallback: pick first band
            img = Image.fromarray(arr[..., 0].astype(np.uint8), mode="L")

        img.save(out_path)

    def _create_preview_with_boxes(
        self,
        input_tif,
        bounding_boxes,
        out_png,
        scale_factor=0.1,
        confidence_threshold=0.999,
        confidences=None,
    ):
        if confidences is not None:
            filtered_boxes, filtered_confidences = [], []
            for box, conf in zip(bounding_boxes, confidences):
                if conf >= confidence_threshold:
                    filtered_boxes.append(box)
                    filtered_confidences.append(conf)
            bounding_boxes, confidences = filtered_boxes, filtered_confidences
            print(
                f"Using {len(bounding_boxes)} boxes with confidence >= {confidence_threshold}"
            )
        with rasterio.open(input_tif) as src:
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)
            resampling_method = Resampling.bilinear
            data = read_raster_data(
                src, out_shape=(new_height, new_width), resampling=resampling_method
            )

            _create_georeferenced_plot(
                raster_array=data.squeeze(),
                src=src,
                title=f"Preview with Merged Bounding Boxes (conf ≥ {confidence_threshold})",
                out_png=Path(out_png),
                bounding_boxes=bounding_boxes,
                add_colorbar=False,
            )

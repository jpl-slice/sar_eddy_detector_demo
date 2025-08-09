import abc
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
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
                            self._plot_and_save_individual_preview(
                                data=window_data,
                                src_transform=src.transform,
                                window=window,
                                bbox=bbox,
                                confidence=confidence,
                                output_path=output_path,
                            )
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
        Given a lon/lat bbox (lon_min, lat_min, lon_max, lat_max),
        reproject it into the src CRS, compute the pixel window,
        buffer it, clamp to the image, and ensure at least patch_size.
        """
        try:
            # 1. Unpack longitude/latitude bounds
            lon_min, lat_min, lon_max, lat_max = bbox

            # 2. Reproject from WGS84 → src.crs (if needed)
            src_crs = src.crs or rasterio.crs.CRS.from_epsg(4326)
            if src_crs.to_epsg() != 4326:
                xs, ys = rasterio.warp.transform(
                    rasterio.crs.CRS.from_epsg(4326),
                    src_crs,
                    [lon_min, lon_max],
                    [lat_max, lat_min],
                )
                x0, x1 = xs
                y0, y1 = ys
            else:
                x0, x1 = lon_min, lon_max
                y0, y1 = (
                    lat_max,
                    lat_min,
                )  # note: index wants (x, y) pairs as (col, row)

            # 3. Convert reprojected corner coords into pixel indices
            col0, row0 = src.index(x0, y0)  # top‐left
            col1, row1 = src.index(x1, y1)  # bottom‐right

            # 4. Add buffer & sort
            row_start, row_end = sorted([row0 - buffer, row1 + buffer])
            col_start, col_end = sorted([col0 - buffer, col1 + buffer])

            # 5. Clamp to image bounds
            row_start = max(0, row_start)
            col_start = max(0, col_start)
            row_end = min(src.height, row_end)
            col_end = min(src.width, col_end)

            row_start, row_end = self._ensure_min_patch_size(
                row_start, row_end, patch_size, src.height
            )
            col_start, col_end = self._ensure_min_patch_size(
                col_start, col_end, patch_size, src.width
            )

            # 6. Check for valid window
            if row_end <= row_start or col_end <= col_start:
                return None

            # 7. Return a Window built from row/col slices
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

    def _plot_and_save_individual_preview(
        self,
        data: np.ndarray,
        src_transform,
        window: Window,
        bbox: Tuple,
        confidence: float,
        output_path: Path,
    ) -> None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        try:
            window_transform = rasterio.windows.transform(window, src_transform)
            left_plot, top_plot = window_transform * (0, 0)
            right_plot, bottom_plot = window_transform * (window.width, window.height)

            # Handle both normalized (uint8) and non-normalized (float) data
            is_normalized_uint8 = data.dtype == np.uint8
            vmax = 255 if is_normalized_uint8 else float(np.nanpercentile(data, 98))
            vmin = 0 if is_normalized_uint8 else float(np.nanmin(data))

            ax.imshow(
                data,
                cmap="gray",
                extent=(left_plot, right_plot, bottom_plot, top_plot),
                vmin=vmin,
                vmax=vmax,
            )
            ax.ticklabel_format(useOffset=False, style="plain")
            # ax.set_title(f"Confidence: {confidence:.3f}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # set axis off
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        except Exception as e:
            print(f"[{self.class_name}] Error plotting/saving preview: {e}")
            traceback.print_exc()
        finally:
            plt.close(fig)

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

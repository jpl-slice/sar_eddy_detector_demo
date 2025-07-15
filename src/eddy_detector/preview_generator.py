import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from matplotlib import patches
from matplotlib import pyplot as plt
from rasterio.windows import Window

from src.utils import parse_bbox
from src.visualize_eddy_bbox import create_preview_with_boxes


class PreviewGenerator:
    def __init__(self, config, detector):
        self.config = config
        self.detector = detector
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_and_prepare_detections_df(
        self, merged: bool, confidence_threshold: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """Loads and prepares the detection DataFrame from CSV for preview generation."""
        bbox_col_name = "bbox"  # Standard column name

        if merged:
            csv_path = self.detector._merged_csv_path
            source_desc = "merged"
        else:
            csv_path = self.detector._base_csv_path
            source_desc = "individual"

        self.logger.info(f"Loading {source_desc} detections from {csv_path}...")

        try:
            df = pd.read_csv(csv_path)  # type: ignore
        except (FileNotFoundError, TypeError):
            self.logger.warning(f"File not found: {csv_path}")
            return None
        except pd.errors.EmptyDataError:
            self.logger.warning(f"No data in CSV file: {csv_path}")
            return None

        if df.empty:
            self.logger.info(
                f"No detections found in {csv_path}. Skipping preview generation."
            )
            return None

        # Ensure the standard bounding box column exists.
        if bbox_col_name not in df.columns:
            self.logger.error(
                f"Required column '{bbox_col_name}' not found in {csv_path}"
            )
            return None

        try:
            # Parse the bounding box string into a tuple of floats
            df["bbox_parsed"] = df[bbox_col_name].apply(parse_bbox)
        except Exception as e:
            self.logger.error(f"Error parsing bbox column in {csv_path}: {e}")
            return None

        # Apply confidence threshold if provided
        if confidence_threshold is not None:
            original_count = len(df)
            # Use .copy() to avoid SettingWithCopyWarning
            df = df[df["confidence"] >= confidence_threshold].copy()
            filtered_count = len(df)
            self.logger.info(
                f"Filtered {original_count - filtered_count} detections below confidence threshold {confidence_threshold}"
            )
            if df.empty:
                self.logger.info("No detections remaining after filtering.")
                # return None
        return df

    def create_scene_previews_with_bbox(self, confidence_threshold=0.99, merged=False):
        """
        Create downsampled PNG preview images with bounding boxes drawn.

        Args:
            confidence_threshold: Minimum confidence to include a bounding box.
            merged: Boolean flag to indicate whether to use merged detections or individual ones.
        """
        df = self._load_and_prepare_detections_df(merged, confidence_threshold)
        if df is None:
            return

        output_prefix = f"full_scene_previews{'_merged_bbox' if merged else ''}"
        output_subdir = f"{output_prefix}_{confidence_threshold}"
        output_dir = Path(self.config.output_dir) / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure that the dataset and its preprocessed directory are available
        if not self.detector.dataset or not self.detector.dataset.preprocessed_dir:
            self.logger.error(
                "Dataset or dir containing preprocessed images is required for preview generation."
            )
            return
        # Ensure create_preview_with_boxes is available
        if create_preview_with_boxes is None:
            self.logger.error(
                "'create_preview_with_boxes' utility not available. Skipping scene previews."
            )
            return

        source_image_folder = Path(self.detector.dataset.preprocessed_dir)
        processed_files_count = 0

        # Loop over each unique image file in the detection CSV
        for filename_base in df["filename"].unique():
            file_path = source_image_folder / str(filename_base)
            if not file_path.exists():
                self.logger.warning(
                    f"Source image not found, skipping preview for {filename_base}"
                )
                continue

            # Get all boxes for this file
            file_df = df[df["filename"] == filename_base]
            boxes_for_file = file_df["bbox_parsed"].tolist()
            confidences = file_df["confidence"].astype(float).tolist()
            try:
                # Generate and save the preview
                output_image_path = output_dir / f"{file_path.stem}_preview.png"
                create_preview_with_boxes(
                    str(file_path),
                    boxes_for_file,
                    str(output_image_path),
                    confidences=confidences,
                    confidence_threshold=confidence_threshold,
                    scale_factor=0.1,
                )
                processed_files_count += 1
            except FileNotFoundError:
                self.logger.error(f"Source file not found at {file_path}")
            except Exception:
                self.logger.error(
                    f"Error generating preview for {filename_base}", exc_info=True
                )

        self.logger.info(
            f"Generated {processed_files_count} preview images in {output_dir}"
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
        """
        Create individual preview images for each positive eddy detection.

        Args:
            confidence_threshold: Only detections with confidence at or above this value are used.
            merged: Flag to indicate if merged detections should be used.
            patch_size: Minimum size for the extracted preview patch.
            bbox_buffer_pixels: Number of pixels to extend the bbox in each direction.
            normalize_window: Flag to indicate if the window data should be normalized.
        """
        df = self._load_and_prepare_detections_df(merged, confidence_threshold)
        if df is None:
            return

        # Define the main directory where individual previews will be saved. Use pathlib.
        # Use parent of base CSV path if available, otherwise use output_dir
        parent_dir = (
            self.detector._base_csv_path.parent
            if self.detector._base_csv_path
            else Path(self.config.output_dir)
        )
        main_output_dir = (
            parent_dir / f"positive_detection_tiles_{confidence_threshold}"
        )
        main_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.detector.dataset or not self.detector.dataset.preprocessed_dir:
            self.logger.error(
                "Dataset or dir containing preprocessed images is required for saving individual previews."
            )
            return

        source_image_folder = Path(self.detector.dataset.preprocessed_dir)

        # Group detections by filename.
        grouped_files = df.groupby("filename")
        self.logger.info(
            f"Creating individual previews for {len(df)} detections across {len(grouped_files)} files in {main_output_dir}"
        )

        # Process each file group
        for filename, group in grouped_files:
            if split_scenes_into_folders:
                file_output_dir = main_output_dir / Path(str(filename)).stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                file_output_dir = main_output_dir
            file_path = source_image_folder / str(filename)
            if not file_path.exists():
                self.logger.warning(
                    f"Source image not found, skipping tile saving for {filename}"
                )
                continue

            try:
                with rasterio.open(file_path) as src:
                    is_jp2 = src.driver == "JP2OpenJPEG" or file_path.suffix == ".jp2"
                    processed_count_for_file = 0
                    for i, detection in group.iterrows():
                        bbox = detection["bbox_parsed"]
                        confidence = detection["confidence"]

                        # Define a unique output path for each detection preview.
                        output_path = (
                            file_output_dir
                            / f"{Path(str(filename)).stem}_detection_{i:03d}_conf_{confidence:.3f}.png"
                        )

                        # Calculate the window to read from the source image.
                        try:
                            window = self._calculate_preview_window(
                                src.transform,
                                src.height,
                                src.width,
                                bbox,
                                bbox_buffer_pixels,
                                patch_size,
                            )
                            if window is None:
                                continue

                            # Read and normalize the data from the calculated window.
                            window_data = self._read_and_normalize_window_data(
                                src, window, is_jp2, normalize_window
                            )
                            if window_data is None:
                                continue

                            # Plot and save the individual preview image.
                            self._plot_and_save_individual_preview(
                                data=window_data,
                                src_transform=src.transform,
                                window=window,
                                bbox=bbox,
                                confidence=confidence,
                                output_path=output_path,
                            )
                            processed_count_for_file += 1
                        except Exception as inner_e:
                            self.logger.error(
                                f"Error processing detection index {i} for {filename}: {inner_e}"
                            )
                            # Optionally add traceback here too
                            # traceback.print_exc()
            except Exception:
                self.logger.error(
                    f"Error processing file {filename} for individual previews",
                    exc_info=True,
                )

        self.logger.info(
            f"All individual previews attempt finished. Check logs for errors. Output dir: {main_output_dir}"
        )

    def _calculate_preview_window(
        self, src_transform, src_height, src_width, bbox, buffer, patch_size
    ) -> Optional[Window]:
        """Calculates the rasterio window for an individual preview."""
        try:
            # Convert geographic bbox to pixel coordinates
            left, bottom, right, top = bbox

            # Inverse transform converts geographic to image (col, row) coordinates.
            col_start, row_start = ~src_transform * (left, top)
            col_end, row_end = ~src_transform * (right, bottom)

            # Ensure coordinates are sorted and add buffer
            row_start, row_end = sorted(
                [int(row_start) - buffer, int(row_end) + buffer]
            )
            col_start, col_end = sorted(
                [int(col_start) - buffer, int(col_end) + buffer]
            )

            # Clip to image boundaries
            row_start, col_start = max(0, row_start), max(0, col_start)
            row_end, col_end = min(src_height, row_end), min(src_width, col_end)

            # Ensure minimum patch size
            row_start, row_end = self._ensure_min_patch_size(
                row_start, row_end, patch_size, src_height
            )
            col_start, col_end = self._ensure_min_patch_size(
                col_start, col_end, patch_size, src_width
            )

            window_height, window_width = row_end - row_start, col_end - col_start
            if window_height <= 0 or window_width <= 0:
                self.logger.warning(
                    f"Calculated invalid window dimensions ({window_width}x{window_height}) for bbox {bbox}. Skipping."
                )
                return None
            return Window(col_start, row_start, window_width, window_height)
        except Exception as e:
            self.logger.error(f"Error calculating preview window: {e}")
            return None

    def _ensure_min_patch_size(self, start, end, patch_size, max_size):
        """Ensure minimum patch size while respecting boundaries."""
        if end - start >= patch_size:
            return start, end

        # Try to center the expansion
        extra = patch_size - (end - start)
        start = max(0, start - extra // 2)
        end = min(max_size, start + patch_size)
        start = max(0, end - patch_size)
        return start, end

    def _read_and_normalize_window_data(
        self, src: rasterio.DatasetReader, window: Window, is_jp2: bool, normalize: bool
    ) -> Optional[np.ndarray]:
        """Reads data for a given window and optionally normalizes it."""
        try:
            if is_jp2:
                # Ensure dataset and expand_from_jp2 are available
                if not self.detector.dataset or not hasattr(
                    self.detector.dataset, "expand_from_jp2"
                ):
                    self.logger.error(
                        "JP2 file detected but dataset or expand_from_jp2 method unavailable."
                    )
                    return None
                data = self.detector.dataset.expand_from_jp2(src, window=window)
            else:
                data = src.read(1, window=window).astype(np.float32)

            # Replace nodata values with NaN
            nodata_val = (
                src.nodata if src.nodata is not None else -9999
            )  # Use a common default if not set
            data[data == nodata_val] = np.nan

            # Optionally normalize
            if normalize:
                with np.errstate(
                    invalid="ignore"
                ):  # Ignore warnings from all-NaN slices
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)
                if (
                    not np.isnan(data_min)
                    and not np.isnan(data_max)
                    and data_max > data_min
                ):
                    data = (data - data_min) / (data_max - data_min)
                elif not np.isnan(
                    data_min
                ):  # Handle case where max == min (flat data)
                    data.fill(0.0)  # Normalize to 0 if flat and not NaN
                # else: data remains NaN if all NaN

            return data
        except Exception:
            self.logger.error("Error reading window data", exc_info=True)
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
        """Plots and saves an individual detection preview."""
        try:
            # Compute window transform and extent for plotting
            window_transform = rasterio.windows.transform(window, src_transform)
            left_plot, top_plot = window_transform * (0, 0)
            right_plot, bottom_plot = window_transform * (window.width, window.height)

            fig, ax = plt.subplots(1, figsize=(10, 10))
            im = ax.imshow(
                data,
                cmap="gray",
                extent=(left_plot, right_plot, bottom_plot, top_plot),
                vmin=0.0,
                # Adjust vmax based on normalization
                vmax=float(
                    1.0 if np.nanmax(data) <= 1.0 else np.nanpercentile(data, 98)
                ),
            )
            ax.ticklabel_format(useOffset=False, style="plain")

            # Bbox coordinates in the window's coordinate system
            # Calculate bbox coordinates relative to the plotted window
            # left, bottom, right, top = bbox
            # bbox_patch_left = max(left, left_plot)
            # bbox_patch_right = min(right, right_plot)
            # bbox_patch_top = min(top, top_plot)  # Geographic top is min row for plot
            # bbox_patch_bottom = max(
            #     bottom, bottom_plot
            # )  # Geographic bottom is max row for plot

            # # Draw the bounding box rectangle
            # rect = patches.Rectangle(
            #     (bbox_patch_left, bbox_patch_bottom),  # Bottom-left corner
            #     bbox_patch_right - bbox_patch_left,  # Width
            #     bbox_patch_top - bbox_patch_bottom,  # Height
            #     linewidth=2,
            #     edgecolor="r",
            #     facecolor="none",
            # )
            # ax.add_patch(rect)

            # Set plot titles and labels
            ax.set_title(f"Confidence: {confidence:.3f}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        except Exception:
            self.logger.error("Error plotting/saving preview", exc_info=True)
        finally:
            plt.close(fig)  # Ensure figure is closed even if error occurs

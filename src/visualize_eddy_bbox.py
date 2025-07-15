import os
import sys

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show

from utils.bbox import boxes_overlap, merge_boxes, parse_bbox
from utils.compress_sar_with_jpeg2000 import expand_from_jp2

# set matplotlib default font size to 24
matplotlib.rcParams.update({"font.size": 24})


def create_preview_with_boxes(
    input_tif,
    bounding_boxes,
    out_png,
    scale_factor=0.1,
    confidence_threshold=0.999,
    confidences=None,
):
    """
    Create a downsampled PNG preview of a GeoTIFF and draw bounding boxes.

    :param input_tif: Path to the large GeoTIFF.
    :param bounding_boxes: List of bounding boxes in lat/lon [(xmin, ymin, xmax, ymax), ...].
    :param out_png: Output path for the PNG preview.
    :param scale_factor: How much to downsample (0.1 => 10% of original).
    :param confidence_threshold: Minimum confidence score to include (default: 0.999).
    :param confidences: List of confidence values corresponding to bounding_boxes.
    """
    # Filter bounding boxes by confidence if provided
    if confidences is not None:
        filtered_boxes = []
        filtered_confidences = []
        for box, conf in zip(bounding_boxes, confidences):
            if conf >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_confidences.append(conf)
        bounding_boxes = filtered_boxes
        confidences = filtered_confidences
        print(
            f"Using {len(bounding_boxes)} boxes with confidence >= {confidence_threshold}"
        )

    with rasterio.open(input_tif) as src:
        # Calculate the new dimensions
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)

        # Read the data at reduced resolution
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)

        if input_tif.endswith("jp2") or src.driver == "JP2OpenJPEG":
            # Expand the data if it was compressed with JPEG2000
            data = expand_from_jp2(
                input_tif,
                out_shape=(new_height, new_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )

        # Adjust the transform so pixel coordinates match the downsampled data
        # (rasterio.transform.scale expects how many *original* pixels each new pixel covers)
        scale_x = src.width / float(new_width)
        scale_y = src.height / float(new_height)
        new_transform = src.transform * src.transform.scale(scale_x, scale_y)

        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 8))
        show(
            data,
            transform=new_transform,
            ax=ax,
            cmap="gray",
            vmin=0,
            vmax=np.nanpercentile(data, 98),
        )  # or any other colormap

        # For each bounding box in lat/lon, draw directly using geographic coordinates
        for i, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
            # Create a width and height for the rectangle
            width = xmax - xmin
            height = ymax - ymin

            # Create rectangle with geographic coordinates directly
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Add confidence label if available
            # if confidences is not None:
            #     plt.text(xmin, ymin, f"{confidences[i]:.3f}",
            #              color='red', fontsize=8, backgroundcolor='white')

        plt.title(f"Preview with Merged Bounding Boxes (conf ≥ {confidence_threshold})")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()


def generate_scene_previews_with_bboxes(
    detector, confidence_threshold=0.99, merged=False
):
    """
    Create downsampled PNG preview images with bounding boxes drawn.

    Args:
        detector: The eddy detector instance.
        confidence_threshold: Minimum confidence to include a bounding box.
        merged: Boolean flag to indicate whether to use merged detections or individual ones.
    """
    df = _load_and_prepare_detections_df(
        detector, merged, confidence_threshold
    )
    if df is None:
        return

    output_prefix = f"full_scene_previews{'_merged_bbox' if merged else ''}"
    output_subdir = f"{output_prefix}_{confidence_threshold}"
    output_dir = Path(detector.config.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not detector.dataset or not detector.dataset.preprocessed_dir:
        detector.logger.error(
            "Dataset or dir containing preprocessed images is required for preview generation."
        )
        return

    source_image_folder = Path(detector.dataset.preprocessed_dir)
    processed_files_count = 0

    for filename_base in df["filename"].unique():
        file_path = source_image_folder / str(filename_base)
        if not file_path.exists():
            detector.logger.warning(
                f"Source image not found, skipping preview for {filename_base}"
            )
            continue

        file_df = df[df["filename"] == filename_base]
        boxes_for_file = file_df["bbox_parsed"].tolist()
        confidences = file_df["confidence"].astype(float).tolist()
        try:
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
            detector.logger.error(f"Source file not found at {file_path}")
        except Exception:
            detector.logger.error(
                f"Error generating preview for {filename_base}", exc_info=True
            )

    detector.logger.info(
        f"Generated {processed_files_count} preview images in {output_dir}"
    )


def save_positive_detection_tiles(
    detector,
    confidence_threshold=0.9,
    merged=True,
    patch_size=128,
    bbox_buffer_pixels=32,
    normalize_window=True,
    split_scenes_into_folders=True,
):
    """
    Create individual preview images for each positive eddy detection.

    Args:
        detector: The eddy detector instance.
        confidence_threshold: Only detections with confidence at or above this value are used.
        merged: Flag to indicate if merged detections should be used.
        patch_size: Minimum size for the extracted preview patch.
        bbox_buffer_pixels: Number of pixels to extend the bbox in each direction.
        normalize_window: Flag to indicate if the window data should be normalized.
    """
    df = _load_and_prepare_detections_df(
        detector, merged, confidence_threshold
    )
    if df is None:
        return

    parent_dir = (
        detector._base_csv_path.parent
        if detector._base_csv_path
        else Path(detector.config.output_dir)
    )
    main_output_dir = (
        parent_dir / f"positive_detection_tiles_{confidence_threshold}"
    )
    main_output_dir.mkdir(parents=True, exist_ok=True)

    if not detector.dataset or not detector.dataset.preprocessed_dir:
        detector.logger.error(
            "Dataset or dir containing preprocessed images is required for saving individual previews."
        )
        return

    source_image_folder = Path(detector.dataset.preprocessed_dir)

    grouped_files = df.groupby("filename")
    detector.logger.info(
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
            detector.logger.warning(
                f"Source image not found, skipping tile saving for {filename}"
            )
            continue

        try:
            with rasterio.open(file_path) as src:
                is_jp2 = src.driver == "JP2OpenJPEG" or file_path.suffix == ".jp2"
                for i, detection in group.iterrows():
                    bbox = detection["bbox_parsed"]
                    confidence = detection["confidence"]

                    output_path = (
                        file_output_dir
                        / f"{Path(str(filename)).stem}_detection_{i:03d}_conf_{confidence:.3f}.png"
                    )

                    try:
                        window = _calculate_preview_window(
                            src.transform,
                            src.height,
                            src.width,
                            bbox,
                            bbox_buffer_pixels,
                            patch_size,
                        )
                        if window is None:
                            continue

                        window_data = _read_and_normalize_window_data(
                            detector, src, window, is_jp2, normalize_window
                        )
                        if window_data is None:
                            continue

                        _plot_and_save_individual_preview(
                            window_data,
                            src.transform,
                            window,
                            bbox,
                            confidence,
                            output_path,
                        )
                    except Exception as inner_e:
                        detector.logger.error(
                            f"Error processing detection index {i} for {filename}: {inner_e}"
                        )
        except Exception:
            detector.logger.error(
                f"Error processing file {filename} for individual previews",
                exc_info=True,
            )

    detector.logger.info(
        f"All individual previews attempt finished. Check logs for errors. Output dir: {main_output_dir}"
    )


def _load_and_prepare_detections_df(
    detector, merged: bool, confidence_threshold: Optional[float] = None
) -> Optional[pd.DataFrame]:
    """Loads and prepares the detection DataFrame from CSV for preview generation."""
    bbox_col_name = "bbox"

    if merged:
        csv_path = detector._merged_csv_path
        source_desc = "merged"
    else:
        csv_path = detector._base_csv_path
        source_desc = "individual"

    detector.logger.info(f"Loading {source_desc} detections from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, TypeError):
        detector.logger.warning(f"File not found: {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        detector.logger.warning(f"No data in CSV file: {csv_path}")
        return None

    if df.empty:
        detector.logger.info(
            f"No detections found in {csv_path}. Skipping preview generation."
        )
        return None

    if bbox_col_name not in df.columns:
        detector.logger.error(
            f"Required column '{bbox_col_name}' not found in {csv_path}"
        )
        return None

    try:
        df["bbox_parsed"] = df[bbox_col_name].apply(parse_bbox)
    except Exception as e:
        detector.logger.error(f"Error parsing bbox column in {csv_path}: {e}")
        return None

    if confidence_threshold is not None:
        original_count = len(df)
        df = df[df["confidence"] >= confidence_threshold].copy()
        filtered_count = len(df)
        detector.logger.info(
            f"Filtered {original_count - filtered_count} detections below confidence threshold {confidence_threshold}"
        )
        if df.empty:
            detector.logger.info("No detections remaining after filtering.")
    return df


def _calculate_preview_window(
    src_transform, src_height, src_width, bbox, buffer, patch_size
) -> Optional[Window]:
    """Calculates the rasterio window for an individual preview."""
    try:
        left, bottom, right, top = bbox
        col_start, row_start = ~src_transform * (left, top)
        col_end, row_end = ~src_transform * (right, bottom)
        row_start, row_end = sorted(
            [int(row_start) - buffer, int(row_end) + buffer]
        )
        col_start, col_end = sorted(
            [int(col_start) - buffer, int(col_end) + buffer]
        )
        row_start, col_start = max(0, row_start), max(0, col_start)
        row_end, col_end = min(src_height, row_end), min(src_width, col_end)
        row_start, row_end = _ensure_min_patch_size(
            row_start, row_end, patch_size, src_height
        )
        col_start, col_end = _ensure_min_patch_size(
            col_start, col_end, patch_size, src_width
        )
        window_height, window_width = row_end - row_start, col_end - col_start
        if window_height <= 0 or window_width <= 0:
            return None
        return Window(col_start, row_start, window_width, window_height)
    except Exception:
        return None


def _ensure_min_patch_size(start, end, patch_size, max_size):
    """Ensure minimum patch size while respecting boundaries."""
    if end - start >= patch_size:
        return start, end
    extra = patch_size - (end - start)
    start = max(0, start - extra // 2)
    end = min(max_size, start + patch_size)
    start = max(0, end - patch_size)
    return start, end


def _read_and_normalize_window_data(
    detector, src: rasterio.DatasetReader, window: Window, is_jp2: bool, normalize: bool
) -> Optional[np.ndarray]:
    """Reads data for a given window and optionally normalizes it."""
    try:
        if is_jp2:
            if not detector.dataset or not hasattr(
                detector.dataset, "expand_from_jp2"
            ):
                detector.logger.error(
                    "JP2 file detected but dataset or expand_from_jp2 method unavailable."
                )
                return None
            data = detector.dataset.expand_from_jp2(src, window=window)
        else:
            data = src.read(1, window=window).astype(np.float32)

        nodata_val = src.nodata if src.nodata is not None else -9999
        data[data == nodata_val] = np.nan

        if normalize:
            with np.errstate(invalid="ignore"):
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
            if (
                not np.isnan(data_min)
                and not np.isnan(data_max)
                and data_max > data_min
            ):
                data = (data - data_min) / (data_max - data_min)
            elif not np.isnan(data_min):
                data.fill(0.0)
        return data
    except Exception:
        detector.logger.error("Error reading window data", exc_info=True)
        return None


def _plot_and_save_individual_preview(
    data: np.ndarray,
    src_transform,
    window: Window,
    bbox: Tuple,
    confidence: float,
    output_path: Path,
):
    """Plots and saves an individual detection preview."""
    try:
        window_transform = rasterio.windows.transform(window, src_transform)
        left_plot, top_plot = window_transform * (0, 0)
        right_plot, bottom_plot = window_transform * (window.width, window.height)

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(
            data,
            cmap="gray",
            extent=(left_plot, right_plot, bottom_plot, top_plot),
            vmin=0.0,
            vmax=float(
                1.0 if np.nanmax(data) <= 1.0 else np.nanpercentile(data, 98)
            ),
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_title(f"Confidence: {confidence:.3f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception:
        plt.close(fig)


if __name__ == "__main__":
    # Example usage
    csv_file = sys.argv[1]
    tif_file = sys.argv[2]
    output_dir = os.path.join(os.path.dirname(csv_file), "previews_0.9")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    df["bbox"] = df["bbox"].apply(parse_bbox)

    file_df = df[df["filename"] == os.path.basename(tif_file)]

    bounding_boxes = file_df["bbox"].tolist()
    confidences = file_df["confidence"].tolist()

    output_png = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(tif_file))[0]}.png"
    )
    print(f"Creating eddy bounding box preview for {tif_file} at {output_png}")

    # Pass both bounding boxes and confidences to the function
    create_preview_with_boxes(
        tif_file,
        bounding_boxes,
        output_png,
        confidences=confidences,
        confidence_threshold=0.9,
    )

import abc
import importlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from rasterio import windows
from rasterio.windows import Window
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from src.dataset import SARTileDataset
from src.utils import load_class, merge_csv_bboxes, parse_bbox
from src.visualize_eddy_bbox import create_preview_with_boxes


class BaseEddyDetector(abc.ABC):
    """
    Abstract Base Class for eddy detection workflows.
    Contains setup, inference, and result handling logic shared by all detector implementations.
    This follows the Template Method design pattern, where subclasses implement specific model and pipeline logic.
    """

    def __init__(self, config):
        """
        Initialize the eddy detector with configuration parameters.

        Args:
            config: Configuration parameters object containing device, paths, and thresholds.
        """
        self.config = config
        self.class_name = self.__class__.__name__  # Alias for logging

        # Set the device: use the provided device if CUDA is available; otherwise default to CPU
        self.device = torch.device(
            config.device
            if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        print(f"[{self.class_name}] Using device: {self.device}")

        # fmt: off
        self.model: Optional[torch.nn.Module] = None  # PyTorch model (e.g., end-to-end or feature extractor)
        self.transform: Optional[transforms.Compose] = None  # Transform pipeline for input preprocessing
        self.dataset: Optional[SARTileDataset] = None
        self.positive_detections: List[Dict] = []  # List that will store detections with high confidence

        # Attributes for model input dimensions and interpolation mode, to be set in subclass setup.
        self.input_size: Optional[Tuple[int, int]] = None
        self.interpolation_mode: Optional[str] = None

        # Paths to generated result files (set after saving)
        self._base_csv_path: Optional[Path] = None
        self._merged_csv_path: Optional[Path] = None

    @abc.abstractmethod
    def _setup_model(self) -> bool:
        """
        Abstract method for subclasses to set up their specific model(s) and any
        related components (like a scikit-learn pipeline).

        This method should populate `self.model` and any other necessary attributes.

        Returns:
            True if setup is successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    def _create_transform(self) -> Optional[transforms.Compose]:
        """
        Abstract method for subclasses to create the specific transformation pipeline
        required for their model.

        Returns:
            A torchvision.transforms.Compose object or None if creation fails.
        """
        pass

    @abc.abstractmethod
    def _predict_batch(
        self, images: torch.Tensor
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Abstract method for subclasses to perform prediction on a batch of images.

        Args:
            images (torch.Tensor): A batch of preprocessed image tensors.

        Returns:
            A tuple containing:
              - predictions (np.ndarray): Array of predicted class indices.
              - probabilities (np.ndarray): Confidence scores for the predicted class.
            Returns (None, None) on failure.
        """
        pass

    def setup(self) -> bool:
        """
        Perform complete setup: load model/pipeline, create transforms, and initialize the dataset.
        Returns:
            True if all setup steps succeed, otherwise False.
        """
        steps = [
            ("model", self._setup_model),
            ("transforms", self._create_and_assign_transform),
            ("dataset", self.setup_dataset),
        ]
        cls_name = self.__class__.__name__
        for name, step_func in steps:
            print(f"[{cls_name}] Setting up {name}...")
            if not step_func():
                print(f"[{cls_name}] {name.capitalize()} setup failed.")
                return False
        return True

    def _create_and_assign_transform(self) -> bool:
        self.transform = self._create_transform()
        return self.transform is not None

    def setup_dataset(self) -> bool:
        """
        Dynamically imports and initializes the dataset class specified in the config.
        Returns:
            True if dataset is set up successfully; False otherwise.
        """
        if self.transform is None:
            print(
                f"[{self.class_name}] Error: Transform must be created before setting up dataset."
            )
            return False

        try:
            data_config = self.config.dataset_params
            class_name = data_config["dataset_class_name"]
            DatasetClass = load_class(class_name, default_pkg="src.dataset")
            # Instantiate the dataset with its specific config and the shared transform
            # Note that the API for any dataset class is: __init__(self, config: Dict, transform: Optional[Callable] = None)
            self.dataset = DatasetClass(data_config, transform=self.transform)
            print(f"[{self.class_name}] Successfully created dataset: {class_name}")
            return True
        except (ImportError, AttributeError) as e:
            print(
                f"[{self.class_name}] Error: Could not import or find dataset class {class_name}. Details: {e}"
            )
            return False
        except Exception as e:
            print(
                f"[{self.class_name}] Error: Failed to initialize dataset. Details: {e}"
            )
            traceback.print_exc()
            return False

    def run_inference(self) -> None:
        """
        Run the detection inference on the dataset, process batches, and save results.
        """
        # Ensure dataset has been initialized.
        if self.dataset is None:
            print(f"[{self.class_name}] Dataset not initialized. Aborting inference.")
            return

        # Create a DataLoader for batching the dataset, with settings based on configuration.
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            prefetch_factor=4 if self.config.workers > 0 else None,
            pin_memory=self.device.type == "cuda",
        )

        # Create output directory using Path for consistent file path handling.
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{self.class_name}] Running inference...")
        # Reset any previous detections.
        self.positive_detections = []
        # Wrap dataloader with tqdm for progress reporting.
        batch_iterator = tqdm(dataloader, desc=f"Processing tiles ({self.class_name})")

        # Ensure no gradients are computed during inference.
        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_iterator):
                images = batch["image"].to(self.device)

                # Get model predictions and their corresponding probabilities
                predictions, probabilities = self._predict_batch(images)

                # If prediction fails for a batch, skip it
                if predictions is None or probabilities is None:
                    print(
                        f"Warning: Skipping batch {batch_idx} due to prediction error."
                    )
                    continue
                # Process each result in the batch
                for i in range(len(predictions)):
                    metadata = {
                        key: val[i] for key, val in batch.items() if key != "image"
                    }
                    self._handle_detection(predictions[i], probabilities[i], metadata)

        # Save the accumulated positive detection results and attempt to merge bounding boxes
        self._save_detection_results()

        # Close the dataset's file handles if a close method is available
        if hasattr(self.dataset, "close"):
            self.dataset.close()

    def _handle_detection(
        self,
        prediction: int,
        probability: float,
        metadata: Dict[str, Any],
        # Removed batch_idx, sample_idx, image, output_dir as they weren't used
    ) -> None:
        """
        Process and store a single detection result if it is positive.

        Args:
            prediction: Predicted class index.
            probability: Confidence score of the predicted class.
            metadata: Additional metadata including filename and bounding box.
        """
        # Only handle detections that correspond to the positive class.
        if prediction != self.config.positive_class_index:
            return

        # Extract filename and bounding box from metadata. Use a default if missing
        # Ensure filename is just the basename
        filename = Path(metadata.get("filename", "unknown_file")).name
        bbox_tensor = metadata.get("bbox", torch.zeros(4))
        # Convert bbox to numpy array if necessary.
        bbox = (
            bbox_tensor.cpu().numpy()
            if isinstance(bbox_tensor, torch.Tensor)
            else np.array(bbox_tensor)
        )
        # Append the detection details to the positive detections list
        self.positive_detections.append(
            {
                "filename": filename,
                "bbox": bbox,
                "confidence": float(probability),
            }
        )

    def _save_detection_results(self) -> None:
        """
        Save individual and merged detection results to CSV files.
        Sets internal attributes _base_csv_path and _merged_csv_path.
        """
        # Ensure the output directory exists.
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct the base CSV path using the configured identification_table_path filename
        base_csv_filename = Path(self.config.identification_table_path).name
        self._base_csv_path = output_dir / base_csv_filename
        self._merged_csv_path = None  # Reset merged path initially

        # Standard column name for bbox string representation in CSV
        bbox_col_name = "bbox"

        # If no positive detections were found, save empty CSVs
        if not self.positive_detections:
            print(f"[{self.class_name}] No positive detections found.")
            # Use standard bbox column name for empty files too
            df = pd.DataFrame(columns=["filename", bbox_col_name, "confidence"])
            df.to_csv(self._base_csv_path, index=False)

            self._merged_csv_path = self._base_csv_path.with_name(
                self._base_csv_path.stem + "_merged.csv"
            )
            df.to_csv(self._merged_csv_path, index=False)

            print(
                f"[{self.__class__.__name__}] Saved empty detection files to: {self._base_csv_path} and {self._merged_csv_path}"
            )
            return  # Exit after saving empty files

        # Create a DataFrame from the detected results
        df = pd.DataFrame(self.positive_detections)
        # Convert bounding box arrays into space-separated strings under the standard column name.
        df[bbox_col_name] = df["bbox"].apply(lambda x: " ".join(map(str, x)))
        # Save the detection results as a CSV, selecting appropriate columns.
        df[["filename", bbox_col_name, "confidence"]].to_csv(
            self._base_csv_path, index=False
        )
        print(
            f"[{self.class_name}] Saved {len(self.positive_detections)} positive detections to: {self._base_csv_path}"
        )

        # --- Attempt to merge bounding boxes ---
        if merge_csv_bboxes is None:
            print(
                f"[{self.class_name}] 'merge_csv_bboxes' not available. Skipping merge."
            )
            return

        try:
            merged_results = merge_csv_bboxes(self._base_csv_path)
            merged_list = []
            # Process each file's detections to create merged bounding boxes
            for filename, detections in merged_results.items():
                for detection in detections:
                    # Convert bounding box to string for saving
                    bbox_str = " ".join(map(str, detection["bbox"]))
                    merged_list.append(
                        {
                            "filename": filename,
                            bbox_col_name: bbox_str,
                            "confidence": detection["confidence"],
                        }
                    )

            if merged_list:  # only save if merging produced results
                merged_df = pd.DataFrame(merged_list)
                merged_filename = (
                    f"{self._base_csv_path.stem}_merged{self._base_csv_path.suffix}"
                )
                self._merged_csv_path = self._base_csv_path.parent / merged_filename
                merged_df.to_csv(self._merged_csv_path, index=False)
                print(
                    f"[{self.class_name}] Saved {len(merged_df)} merged detections to: {self._merged_csv_path}"
                )
            else:
                print(f"[{self.class_name}] No overlapping boxes to merge.")

        except FileNotFoundError:
            print(
                f"[{self.class_name}] Error: Base CSV file not found for merging at {self._base_csv_path}"
            )
        except Exception as e:
            print(f"[{self.class_name}] Error merging bounding boxes: {e}")
            traceback.print_exc()

    # --- Preview Generation Helper ---

    def _load_and_prepare_detections_df(
        self, merged: bool, confidence_threshold: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """Loads and prepares the detection DataFrame from CSV for preview generation."""
        bbox_col_name = "bbox"  # Standard column name

        if merged:
            csv_path = self._merged_csv_path
            source_desc = "merged"
        else:
            csv_path = self._base_csv_path
            source_desc = "individual"

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

        # Ensure the standard bounding box column exists.
        if bbox_col_name not in df.columns:
            print(
                f"[{self.class_name}] Error: Required column '{bbox_col_name}' not found in {csv_path}"
            )
            return None

        try:
            # Parse the bounding box string into a tuple of floats
            df["bbox_parsed"] = df[bbox_col_name].apply(parse_bbox)
        except Exception as e:
            print(f"[{self.class_name}] Error parsing bbox column in {csv_path}: {e}")
            return None

        # Apply confidence threshold if provided
        if confidence_threshold is not None:
            original_count = len(df)
            # Use .copy() to avoid SettingWithCopyWarning
            df = df[df["confidence"] >= confidence_threshold].copy()
            filtered_count = len(df)
            print(
                f"[{self.__class__.__name__}] Filtered {original_count - filtered_count} detections below confidence threshold {confidence_threshold}"
            )
            if df.empty:
                print(
                    f"[{self.__class__.__name__}] No detections remaining after filtering."
                )
                # return None
        return df

    # --- Main Preview Methods ---

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
        if not self.dataset or not self.dataset.preprocessed_dir:
            print(
                f"[{self.class_name}] Dataset or dir containing preprocessed images is required for preview generation."
            )
            return
        # Ensure create_preview_with_boxes is available
        if create_preview_with_boxes is None:
            print(
                f"[{self.class_name}] Error: 'create_preview_with_boxes' utility not available. Skipping scene previews."
            )
            return

        source_image_folder = Path(self.dataset.preprocessed_dir)
        processed_files_count = 0

        # Loop over each unique image file in the detection CSV
        for filename_base in df["filename"].unique():
            file_path = source_image_folder / str(filename_base)
            if not file_path.exists():
                print(
                    f"[{self.class_name}] Warning: Source image not found, skipping preview for {filename_base}"
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
            self._base_csv_path.parent
            if self._base_csv_path
            else Path(self.config.output_dir)
        )
        main_output_dir = (
            parent_dir / f"positive_detection_tiles_{confidence_threshold}"
        )
        main_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.dataset or not self.dataset.preprocessed_dir:
            print(
                f"[{self.class_name}] Dataset or dir containing preprocessed images is required for saving individual previews."
            )
            return

        source_image_folder = Path(self.dataset.preprocessed_dir)

        # Group detections by filename.
        grouped_files = df.groupby("filename")
        print(
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
                print(
                    f"[{self.class_name}] Warning: Source image not found, skipping tile saving for {filename}"
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
                            print(
                                f"  Error processing detection index {i} for {filename}: {inner_e}"
                            )
                            # Optionally add traceback here too
                            # traceback.print_exc()
            except Exception as e:
                print(
                    f"[{self.class_name}] Error processing file {filename} for individual previews: {e}"
                )
                traceback.print_exc()

        print(
            f"[{self.class_name}] All individual previews attempt finished. Check logs for errors. Output dir: {main_output_dir}"
        )

    # --- Individual Preview Helpers ---

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
                print(
                    f"  Warning: Calculated invalid window dimensions ({window_width}x{window_height}) for bbox {bbox}. Skipping."
                )
                return None
            return Window(col_start, row_start, window_width, window_height)
        except Exception as e:
            print(f"[{self.class_name}] Error calculating preview window: {e}")
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
                if not self.dataset or not hasattr(self.dataset, "expand_from_jp2"):
                    print(
                        "  Error: JP2 file detected but dataset or expand_from_jp2 method unavailable."
                    )
                    return None
                data = self.dataset.expand_from_jp2(src, window=window)
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
                elif not np.isnan(data_min):  # Handle case where max == min (flat data)
                    data.fill(0.0)  # Normalize to 0 if flat and not NaN
                # else: data remains NaN if all NaN

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
            # ax.set_title(f"Confidence: {confidence:.3f}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        except Exception as e:
            print(f"[{self.class_name}] Error plotting/saving preview: {e}")
            traceback.print_exc()
        finally:
            plt.close(fig)  # Ensure figure is closed even if error occurs

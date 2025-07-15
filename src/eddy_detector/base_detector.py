import abc
import importlib
import logging
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
from src.visualize_eddy_bbox import (
    create_preview_with_boxes,
    generate_scene_previews_with_bboxes,
    save_positive_detection_tiles,
)


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
        self.arch = self.config.arch  # Model architecture name
        self.class_name = self.__class__.__name__  # Alias for logging

        # Set the device: use the provided device if CUDA is available; otherwise default to CPU
        self.device = torch.device(
            config.device
            if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")

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
        for name, step_func in steps:
            self.logger.info(f"Setting up {name}...")
            if not step_func():
                self.logger.error(f"{name.capitalize()} setup failed.")
                return False
        return True

    def _setup_model(self) -> bool:
        """
        Boilerplate to load the model. Subclasses should call this and handle any
        additional setup (e.g., pipelines, transforms) as needed.
        """
        self.model, self.input_size, self.interpolation_mode = self._load_model()
        return self.model is not None

    def _load_model(self):
        """
        Dynamically import and initialize the model based on config.
        Returns:
            model, input_size, interpolation_mode on success;
            (None, None, None) on failure.
        """
        path = getattr(self.config, "model_loader_class", None)
        if not path:
            self.logger.error("'model_loader_class' not specified in config.")
            return None, None, None
        try:
            Loader = load_class(path, default_pkg="src.models")
            model, size, interp = Loader.load(self.config)
            return model, size, interp
        except Exception:
            self.logger.error(f"Error loading model '{path}'", exc_info=True)
            return None, None, None

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
            self.logger.error(
                "Transform must be created before setting up dataset."
            )
            return False

        try:
            data_config = self.config.dataset_params
            class_name = data_config["dataset_class_name"]
            DatasetClass = load_class(class_name, default_pkg="src.dataset")
            # Instantiate the dataset with its specific config and the shared transform
            # Note that the API for any dataset class is: __init__(self, config: Dict, transform: Optional[Callable] = None)
            self.dataset = DatasetClass(data_config, transform=self.transform)
            self.logger.info(f"Successfully created dataset: {class_name}")
            return True
        except (ImportError, AttributeError) as e:
            self.logger.error(
                f"Could not import or find dataset class {class_name}. Details: {e}"
            )
            return False
        except Exception:
            self.logger.error("Failed to initialize dataset.", exc_info=True)
            return False

    def run_inference(self) -> None:
        """
        Run the detection inference on the dataset, process batches, and save results.
        """
        # Ensure dataset has been initialized.
        if self.dataset is None:
            self.logger.error("Dataset not initialized. Aborting inference.")
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

        self.logger.info("Running inference...")
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
                    self.logger.warning(
                        f"Skipping batch {batch_idx} due to prediction error."
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
            self.logger.info("No positive detections found.")
            # Use standard bbox column name for empty files too
            df = pd.DataFrame(columns=["filename", bbox_col_name, "confidence"])
            df.to_csv(self._base_csv_path, index=False)

            self._merged_csv_path = self._base_csv_path.with_name(
                self._base_csv_path.stem + "_merged.csv"
            )
            df.to_csv(self._merged_csv_path, index=False)

            self.logger.info(
                f"Saved empty detection files to: {self._base_csv_path} and {self._merged_csv_path}"
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
        self.logger.info(
            f"Saved {len(self.positive_detections)} positive detections to: {self._base_csv_path}"
        )

        # --- Attempt to merge bounding boxes ---
        if merge_csv_bboxes is None:
            self.logger.warning("'merge_csv_bboxes' not available. Skipping merge.")
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
                self.logger.info(
                    f"Saved {len(merged_df)} merged detections to: {self._merged_csv_path}"
                )
            else:
                self.logger.info("No overlapping boxes to merge.")

        except FileNotFoundError:
            self.logger.error(
                f"Base CSV file not found for merging at {self._base_csv_path}"
            )
        except Exception:
            self.logger.error("Error merging bounding boxes", exc_info=True)

    # --- Main Preview Methods ---

    def generate_scene_previews_with_bboxes(
        self, confidence_threshold=0.99, merged=False
    ):
        """
        Create downsampled PNG preview images with bounding boxes drawn.
        Args:
            confidence_threshold: Minimum confidence to include a bounding box.
            merged: Boolean flag to indicate whether to use merged detections or individual ones.
        """
        generate_scene_previews_with_bboxes(self, confidence_threshold, merged)

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
        save_positive_detection_tiles(
            self,
            confidence_threshold,
            merged,
            patch_size,
            bbox_buffer_pixels,
            normalize_window,
            split_scenes_into_folders,
        )

import abc
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from src.dataset import SARTileDataset
from src.utils.bbox import merge_csv_bboxes
from src.utils.importer import load_class
from src.visualize_eddy_bbox import EddyVisualizer


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
        self.visualizer = EddyVisualizer(self)

        # Optional class names for mapping indices to human-readable labels
        self.class_names: Optional[List[str]] = None

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

    def _setup_model(self) -> bool:
        """
        Boilerplate to load the model. Subclasses should call this and handle any
        additional setup (e.g., pipelines, transforms) as needed.
        """
        self.model, self.input_size, self.interpolation_mode = self._load_model()

        # Capture optional class_names from config for downstream labeling
        if hasattr(self.config, "class_names") and self.config.class_names:
            try:
                self.class_names = list(self.config.class_names)
                assert (
                    len(self.class_names) == self.config.num_classes
                ), f"Expected {self.config.num_classes} class names, got {self.class_names}"
            except Exception:
                self.class_names = None
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
            print(
                f"[{self.class_name}] Error: 'model_loader_class' not specified in config."
            )
            return None, None, None
        try:
            Loader = load_class(path, default_pkg="src.models")
            model, size, interp = Loader.load(self.config)
            return model, size, interp
        except Exception as e:
            print(f"[{self.class_name}] Error loading model '{path}': {e}")
            traceback.print_exc()
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
        # Extract filename and bounding box from metadata. Use a default if missing
        # Ensure filename is just the basename
        filename = Path(metadata.get("filename", "unknown_file")).name
        bbox_tensor = metadata.get("bbox_geo", torch.zeros(4))
        # Convert bbox to numpy array if necessary.
        bbox = (
            bbox_tensor.cpu().numpy()
            if isinstance(bbox_tensor, torch.Tensor)
            else np.array(bbox_tensor)
        )
        # Append the detection details to the positive detections list
        row = {
            "filename": filename,
            "bbox": bbox,
            "confidence": float(probability),
        }
        if self.is_multiclass():
            row["pred_class"] = int(prediction)
            if self.class_names and 0 <= int(prediction) < len(self.class_names):
                row["pred_label"] = str(self.class_names[int(prediction)])
        self.positive_detections.append(row)

    def is_multiclass(self):
        return self.config.num_classes is not None and self.config.num_classes > 1

    def _save_detection_results(self) -> None:
        """
        Save individual and merged detection results to CSV files.
        Sets internal attributes _base_csv_path and _merged_csv_path.
        """
        # Ensure the output directory exists.
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_csv_filename = Path(self.config.identification_table_path).name
        self._base_csv_path = output_dir / base_csv_filename

        self._save_raw_detections()
        self._merge_and_save_detections()

    def _save_raw_detections(self, bbox_col_name="bbox") -> None:
        """Saves the raw, unmerged detection results to a CSV file.
        Args:
            bbox_col_name: The column name for bounding box representation in the CSV.
        """

        # If no positive detections were found, save empty CSVs
        if not self.positive_detections:
            self.save_empty_csv(bbox_col_name, self._base_csv_path)
            return

        # Create a DataFrame from the detected results
        df = pd.DataFrame(self.positive_detections)
        # Convert bounding box arrays into space-separated strings under the standard column name.
        df[bbox_col_name] = df["bbox"].apply(lambda x: " ".join(map(str, x)))
        # Save the detection results as a CSV
        df.to_csv(self._base_csv_path, index=False)
        print(
            f"[{self.class_name}] Saved {len(self.positive_detections)} positive detections to: {self._base_csv_path}"
        )

    def save_empty_csv(self, bbox_col_name, csv_filepath):
        print(f"[{self.class_name}] No positive detections found. Saving empty CSV.")
        # Use standard bbox column name for empty files too
        columns = ["filename", bbox_col_name, "confidence"]
        if self.is_multiclass():
            columns.append("pred_class")
        df = pd.DataFrame(columns=columns)  # empty df
        df.to_csv(csv_filepath, index=False)

    def _merge_and_save_detections(self, bbox_col_name="bbox") -> None:
        """Merges bounding boxes from the raw detections CSV and saves them to a new file."""
        # --- Attempt to merge bounding boxes ---
        if merge_csv_bboxes is None:
            print(
                f"[{self.class_name}] 'merge_csv_bboxes' not available. Skipping merge."
            )
            return

        if self._base_csv_path is None:
            print(f"[{self.class_name}] Base CSV path is not set. Skipping merge.")
            return
        self._merged_csv_path = self._base_csv_path.with_name(
            self._base_csv_path.stem + "_merged.csv"
        )

        if not self.positive_detections:
            # Create an empty merged file as well
            self.save_empty_csv(bbox_col_name, self._merged_csv_path)
            return

        try:
            # Extract bbox merging parameters from config
            bbox_config = getattr(self.config, "bbox_merging", {})
            merged_results = merge_csv_bboxes(
                str(self._base_csv_path),
                nms_iou_threshold=bbox_config.get("nms_iou_threshold", 0.05),
                merge_iou_threshold=bbox_config.get("merge_iou_threshold", 0.3),
                post_nms_iou_threshold=bbox_config.get("post_nms_iou_threshold", 0.1),
            )
            merged_list = []
            # Process each file's detections to create merged bounding boxes
            for filename, detections in merged_results.items():
                for detection in detections:
                    # Convert bounding box to string for saving
                    bbox_str = " ".join(map(str, detection.pop("bbox")))
                    merged_list.append(
                        {"filename": filename, bbox_col_name: bbox_str, **detection}
                    )

            if merged_list:  # only save if merging produced results
                merged_df = pd.DataFrame(merged_list)
                # Ensure pred_class column exists for acceptance; fill with NA if absent
                if self.is_multiclass() and "pred_class" not in merged_df.columns:
                    merged_df["pred_class"] = pd.NA
                merged_df.to_csv(self._merged_csv_path, index=False)
                print(
                    f"[{self.class_name}] Saved {len(merged_df)} merged detections to: {self._merged_csv_path}"
                )
            else:  # the block below triggers if `merge_csv_bboxes` somehow returns []
                print(f"[{self.class_name}] No overlapping boxes to merge.")
                self.save_empty_csv(bbox_col_name, self._merged_csv_path)

        except FileNotFoundError:
            print(
                f"[{self.class_name}] Error: Base CSV file not found for merging at {self._base_csv_path}"
            )
        except Exception as e:
            print(f"[{self.class_name}] Error merging bounding boxes: {e}")
            traceback.print_exc()

    def create_scene_previews_with_bbox(self, confidence_threshold=0.99, merged=False):
        self.visualizer.create_scene_previews_with_bbox(confidence_threshold, merged)

    def save_positive_detection_tiles(
        self,
        confidence_threshold=0.9,
        merged=True,
        patch_size=128,
        bbox_buffer_pixels=32,
        normalize_window=True,
        split_scenes_into_folders=True,
    ) -> None:
        self.visualizer.save_positive_detection_tiles(
            confidence_threshold,
            merged,
            patch_size,
            bbox_buffer_pixels,
            normalize_window,
            split_scenes_into_folders,
        )

import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple

import joblib
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

from src.eddy_detector.base_detector import BaseEddyDetector
from src.transforms import ClipNormalizeCastToUint8


class CombinedMaskingTransform:
    def __init__(self, nodata_value, img_transform, mask_transform):
        self.nodata_value = nodata_value
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __call__(self, img):
        arr = np.array(img)
        condition = (arr == self.nodata_value) | np.isnan(arr)
        mask = Image.fromarray(condition.astype(np.uint8) * 255)

        img_transformed = self.img_transform(img)
        mask_transformed = self.mask_transform(mask)

        img_transformed[mask_transformed > 0] = self.nodata_value
        return img_transformed


class TimmXGBoostEddyDetector(BaseEddyDetector):
    """Eddy detector using TIMM feature extractor and scikit-learn pipeline."""

    def _setup_model(self) -> bool:
        """Loads TIMM feature extractor and scikit-learn pipeline."""
        # 1. TIMM model (using args.model_loader_class == TimmLoader)
        if not super()._setup_model():  # sets self.model, input_size, interp_mode
            return False

        # 2. scikit-learn pipeline
        pipeline_path = Path(getattr(self.config, "pipeline_path", ""))
        try:
            self.pipeline = joblib.load(pipeline_path)
            print(
                f"[{self.class_name}] Pipeline loaded: {type(self.pipeline).__name__}"
            )
        except Exception as e:
            print(f"[{self.class_name}] Error loading pipeline: {e}")
            traceback.print_exc()
            return False

        return True

    def _create_transform(self) -> Optional[Callable]:
        """Creates the transform pipeline using TIMM helpers."""
        if self.model is None or self.input_size is None:
            print(
                f"[{self.class_name}] Error: TIMM model must be loaded first to determine transforms."
            )
            return None
        print(
            f"[{self.class_name}] Creating transform pipeline using timm helpers for model: {self.arch}"
        )
        try:
            data_config = timm.data.resolve_model_data_config(self.model)
            transform = timm.data.create_transform(**data_config, is_training=False)
            print(f"  TIMM data config resolved: {data_config}")
            print(f"  TIMM transform pipeline created.")

            default_config = self.model.default_cfg
            # if default_config.get("fixed_input_size", False):
            #     data_config["input_size"] = (data_config["input_size"][0], self.config.window_size, self.config.window_size)
            #     print(f"    This model supports variable input size, so we set it to {data_config['input_size']}.")
            #     print(f"    Updated data config: {data_config}")

            first_two_transforms = [ClipNormalizeCastToUint8(), transforms.ToPILImage()]

            # Check if Grayscale conversion needs to be added PREPENDED
            # Assumes dataset provides PIL Image 'L' mode
            self.prepend_grayscale_if_needed(data_config, transform)
            mask_config = data_config.copy()
            mask_config.update(
                {
                    "interpolation": "nearest",  # Use nearest neighbor to avoid blending.
                    "mean": (0.0,),
                    "std": (1.0,),
                }
            )
            mask_transforms = timm.data.create_transform(
                **mask_config, is_training=False
            )
            self.prepend_grayscale_if_needed(mask_config, mask_transforms)
            transform.transforms.insert(0, ClipNormalizeCastToUint8())
            transform.transforms.insert(1, transforms.ToPILImage())
            print(f"Using the following transforms: {transform.transforms}")
            return CombinedMaskingTransform(
                nodata_value=self.config.nodata_value,
                img_transform=transform,
                mask_transform=mask_transforms,
            )
        except Exception as e:
            print(
                f"[{self.class_name}] Error creating TIMM transform: {e}. Check timm version compatibility."
            )
            traceback.print_exc()
            return None

    def prepend_grayscale_if_needed(self, data_config, transform):
        if data_config.get("input_size", [3])[0] == 3:  # If model expects 3 channels
            has_grayscale = any(
                isinstance(t, transforms.Grayscale) and t.num_output_channels == 3
                for t in transform.transforms
            )
            if not has_grayscale:
                print(
                    "  Prepending Grayscale(num_output_channels=3) to the transform pipeline."
                )
                # Ensure Grayscale is the very first operation on the PIL image
                transform.transforms.insert(
                    0, transforms.Grayscale(num_output_channels=3)
                )  # Indicate failure

    def _predict_batch(
        self, images: torch.Tensor
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts features and uses the scikit-learn pipeline for prediction."""
        if self.model is None or self.pipeline is None:
            print(f"[{self.class_name}] Error: Model and pipeline must be loaded.")
            return None, None
        try:
            # 1. Extract features (runs on self.device)
            features = self.model(images)
            # Move features to CPU for scikit-learn pipeline
            features_np = features.cpu().numpy()

            # 2. Predict using the loaded pipeline (expects CPU numpy array)
            predictions_np = self.pipeline.predict(features_np)

            # 3. Get probabilities
            probabilities_np: Optional[np.ndarray] = None
            if hasattr(self.pipeline, "predict_proba"):
                # Use predict_proba which returns probabilities for all classes
                probabilities_all = self.pipeline.predict_proba(features_np)
                # Extract the probability of the *predicted* class
                probabilities_np = probabilities_all[
                    np.arange(len(predictions_np)), predictions_np
                ]
            else:
                # If predict_proba is not available, assign fixed confidence
                print(
                    f"Warning: Pipeline {type(self.pipeline)} does not have 'predict_proba'. Assigning confidence=1.0."
                )
                probabilities_np = np.ones_like(predictions_np, dtype=float)

            return predictions_np, probabilities_np

        except Exception as e:
            print(f"[{self.class_name}] Error during TIMM+Pipeline prediction: {e}")
            traceback.print_exc()
            return None, None  # Indicate failure

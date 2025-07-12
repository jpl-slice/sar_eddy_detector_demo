import traceback
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from src.eddy_detector.base_detector import BaseEddyDetector
from src.model import get_model
from src.transforms import SARImageNormalizer


class SimCLREddyDetector(BaseEddyDetector):
    """Eddy detector using the original SimCLR ResNet model."""

    def _setup_model(self) -> bool:
        """Loads the SimCLR model and sets input properties."""
        try:
            self.model, self.input_size, self.interpolation_mode = get_model(
                self.config
            )
            if self.model is None:
                return False
            # Model should be on device and in eval mode from get_model
            return True
        except Exception as e:
            print(f"[{self.class_name}] Error loading SimCLR model: {e}")
            traceback.print_exc()
            return False

    def _create_transform(self) -> transforms.Compose:
        """Create the transformation pipeline for SAR imagery. Simplified for demo."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST
                ),
                SARImageNormalizer.global_min_max_normalize,
                SARImageNormalizer.per_tile_normalize,
                SARImageNormalizer.boost_dark_images,
            ]
        )

    def _create_transform_imagenet(self) -> Optional[transforms.Compose]:
        """Creates the transform pipeline for the SimCLR model."""
        print(f"[{self.class_name}] Creating transform pipeline.")
        input_size = self.input_size or (224, 224)  # Use determined size or default
        # Use determined interpolation or default
        interpolation_str = self.interpolation_mode or "bilinear"
        try:
            interpolation = transforms.InterpolationMode(interpolation_str.upper())
        except ValueError:
            print(
                f"Warning: Invalid interpolation mode '{interpolation_str}'. Defaulting to BILINEAR."
            )
            interpolation = transforms.InterpolationMode.BILINEAR

        # Assumes input is PIL Image 'L' mode from dataset __getitem__
        return transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),  # Convert L -> RGB equiv
                transforms.Resize(input_size, interpolation=interpolation),
                transforms.ToTensor(),  # Converts PIL [0,255] to Tensor [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Standard ImageNet norm
            ]
        )

    def _predict_batch(
        self, images: torch.Tensor
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Runs inference using the SimCLR model."""
        try:
            outputs = self.model(images)
            probs_tensor = torch.softmax(outputs, dim=1)
            preds_tensor = torch.argmax(probs_tensor, dim=1)

            # Extract probability of the *predicted* class
            probabilities_np = _get_probabilities(probs_tensor, preds_tensor)
            predictions_np = preds_tensor.cpu().numpy()

            return predictions_np, probabilities_np
        except Exception as e:
            print(f"[{self.class_name}] Error during SimCLR prediction: {e}")
            traceback.print_exc()
            return None, None


def _get_probabilities(
    probs_tensor: torch.Tensor, preds_tensor: torch.Tensor
) -> np.ndarray:
    """
    Extract the probability for the predicted class for each sample.
    """
    # Move tensors to CPU and convert to numpy arrays for further processing
    probs_np = probs_tensor.cpu().numpy()
    preds_np = preds_tensor.cpu().numpy()
    batch_size = probs_np.shape[0]
    # Select the probability corresponding to the predicted index for each sample
    return probs_np[np.arange(batch_size), preds_np]

import logging
import traceback
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from src.eddy_detector.base_detector import BaseEddyDetector
from src.transforms import SARImageNormalizer


class SimCLREddyDetector(BaseEddyDetector):
    """Eddy detector using the original SimCLR ResNet model."""

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
        except (RuntimeError, ValueError) as e:
            self.logger.error(f"Error during SimCLR prediction: {e}", exc_info=True)
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

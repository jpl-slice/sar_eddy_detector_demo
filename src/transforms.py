import numpy as np
import torch
from torchvision import transforms

# --- Normalization constants for SAR data ---
sar_min = 2.341661e-09
sar_max = 1123.6694


class SARImageNormalizer:
    """Handles normalization operations for SAR imagery."""

    @staticmethod
    def global_min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Apply global min-max normalization based on dataset statistics."""
        return transforms.Normalize(mean=sar_min, std=sar_max - sar_min)(tensor)

    @staticmethod
    def per_tile_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Apply per-tile normalization to handle local contrast."""
        # Exclude values of -9999 from the img array
        valid_values = tensor[tensor != -9999]
        if valid_values.numel() == 0:
            return tensor  # Return original if no valid values
        valid_min, valid_max = valid_values.min(), valid_values.max()
        std = valid_max - valid_min
        if std == 0:
            return tensor  # Avoid division by zero
        return transforms.Normalize(mean=valid_min, std=std)(tensor)

    @staticmethod
    def boost_dark_images(tensor: torch.Tensor) -> torch.Tensor:
        """Boost dark images to improve contrast."""
        if tensor.mean() < 1e-5:
            tensor = tensor * 10
        return tensor


class ClipNormalizeCastToUint8(torch.nn.Module):
    """
    Custom transform to clip, normalize, and cast SAR data to uint8 for models
    that expect image-like inputs (e.g., TIMM).
    """

    def forward(self, data, norm_max_percent=99):
        # Clip to the 99th percentile to handle extreme outliers
        p_max = np.nanpercentile(data, norm_max_percent)
        p_min = np.nanmin(data)

        data_clipped = np.clip(data, p_min, p_max)

        # Scale to 1-255, reserving 0 for nodata/NaN
        denominator = p_max - p_min
        if denominator == 0:  # Avoid division by zero for uniform tiles
            data_scaled = np.ones_like(data_clipped) * 128
        else:
            data_scaled = ((data_clipped - p_min) / denominator) * 254.0 + 1.0

        # Set NaNs to 0 after scaling
        data_scaled[np.isnan(data_scaled)] = 0

        data_uint8 = data_scaled.astype(np.uint8).squeeze()
        return data_uint8

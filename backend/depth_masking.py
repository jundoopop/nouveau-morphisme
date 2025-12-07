"""
Depth-Guided Background Masking Module

This module uses Depth-Anything V2 to create high-quality foreground masks
based on depth information. It can be used standalone or combined with
traditional segmentation methods (like rembg) for superior results.

Key advantages of depth-based masking:
- Handles objects with similar colors to background
- Better separation in complex scenes
- More robust to lighting conditions
- Can identify depth-based foreground/background
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Depth-Anything V2 will be imported dynamically to avoid errors if not installed
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_AVAILABLE = True
except ImportError:
    logger.warning("Depth-Anything-V2 not installed. Depth-based masking will be unavailable.")
    DEPTH_AVAILABLE = False


class DepthGuidedMasker:
    """
    Depth-guided masking using Depth-Anything V2.

    This class provides methods to:
    1. Estimate depth maps from images
    2. Create foreground masks based on depth thresholds
    3. Combine depth masks with segmentation masks for hybrid approach
    """

    def __init__(self, device: torch.device, model_size: str = 'vitb'):
        """
        Initialize Depth-Anything V2 model.

        Args:
            device: PyTorch device (cuda, mps, or cpu)
            model_size: Model size variant
                - 'vits': Small, fast (97MB, ~0.3-0.5s)
                - 'vitb': Base, balanced (380MB, ~0.5-1.0s) [RECOMMENDED]
                - 'vitl': Large, best quality (1.3GB, ~1.0-2.0s)

        Raises:
            ImportError: If Depth-Anything-V2 is not installed
            FileNotFoundError: If model checkpoint not found
        """
        if not DEPTH_AVAILABLE:
            raise ImportError(
                "Depth-Anything-V2 is not installed. "
                "Install with: pip install depth-anything-v2"
            )

        self.device = device
        self.model_size = model_size

        # Model configurations
        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384]
            },
            'vitb': {
                'encoder': 'vitb',
                'features': 128,
                'out_channels': [96, 192, 384, 768]
            },
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024]
            }
        }

        if model_size not in model_configs:
            raise ValueError(
                f"Invalid model_size: {model_size}. "
                f"Choose from: {list(model_configs.keys())}"
            )

        config = model_configs[model_size]

        logger.info(f"Initializing Depth-Anything V2 ({model_size})...")

        # Initialize model
        self.model = DepthAnythingV2(**config)

        # Load checkpoint (use path relative to this file, not CWD)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_dir, 'checkpoints', f'depth_anything_v2_{model_size}.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {checkpoint_path}\n"
                f"Download it with:\n"
                f"mkdir -p backend/checkpoints\n"
                f"wget https://huggingface.co/depth-anything/Depth-Anything-V2-{model_size.capitalize()}/resolve/main/depth_anything_v2_{model_size}.pth "
                f"-O {checkpoint_path}"
            )

        # Load weights
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location='cpu')
        )
        self.model.to(device).eval()

        logger.info(f"Depth-Anything V2 ({model_size}) loaded successfully on {device}")

    @torch.no_grad()
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth map from image.

        Args:
            image: Input PIL image

        Returns:
            Normalized depth map (H, W) with values 0-1
            Smaller values = closer (foreground)
            Larger values = farther (background)
        """
        # Convert to RGB numpy array
        image_rgb = np.array(image.convert('RGB'))

        # Infer depth
        depth = self.model.infer_image(image_rgb)

        # Normalize to 0-1 range
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth_normalized

    def create_depth_mask(
        self,
        image: Image.Image,
        threshold_percentile: int = 30,
        morph_kernel_size: int = 7,
        blur_size: int = 7
    ) -> np.ndarray:
        """
        Create foreground mask by keeping close objects based on depth.

        Args:
            image: Input PIL image
            threshold_percentile: Keep pixels closer than this percentile
                - Lower (20-30): Only very close objects
                - Medium (30-40): Typical foreground
                - Higher (40-50): More depth range included
            morph_kernel_size: Morphological operation kernel size
            blur_size: Edge smoothing blur size

        Returns:
            Binary mask (H, W) with values 0-255
        """
        logger.debug(f"Creating depth mask with threshold_percentile={threshold_percentile}")

        # Get depth map
        depth_normalized = self.estimate_depth(image)

        # Create mask: keep closest objects (smallest depth values)
        threshold = np.percentile(depth_normalized, threshold_percentile)
        mask = (depth_normalized < threshold).astype(np.uint8) * 255

        logger.debug(f"Depth threshold: {threshold:.3f}, "
                    f"foreground pixels: {(mask > 0).sum()}/{mask.size}")

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )

        # Fill holes (close gaps in foreground)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Remove noise (remove small background blobs)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Smooth edges
        if blur_size > 0:
            if blur_size % 2 == 0:
                blur_size += 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask

    def hybrid_mask(
        self,
        image: Image.Image,
        segmentation_mask: np.ndarray,
        depth_percentile: int = 40,
        strategy: str = "intersection"
    ) -> np.ndarray:
        """
        Combine depth mask with segmentation mask for best results.

        Strategy options:
        - "intersection" (AND): Both depth and segmentation must agree
          → Most conservative, cleanest results, may cut into object
        - "union" (OR): Either depth or segmentation
          → Most inclusive, may include background
        - "depth_guided": Use depth to refine segmentation edges
          → Best balance, recommended

        Args:
            image: Input PIL image
            segmentation_mask: Binary mask from rembg/other (H, W) 0-255
            depth_percentile: Depth threshold percentile
            strategy: Combination strategy

        Returns:
            Combined binary mask (H, W) 0-255
        """
        logger.debug(f"Creating hybrid mask with strategy='{strategy}'")

        # Get depth mask
        depth_mask = self.create_depth_mask(
            image,
            threshold_percentile=depth_percentile,
            morph_kernel_size=7,
            blur_size=5
        )

        if strategy == "intersection":
            # Both must agree (conservative)
            combined = cv2.bitwise_and(depth_mask, segmentation_mask)

            # Fill small holes that might have been created
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        elif strategy == "union":
            # Either can include (inclusive)
            combined = cv2.bitwise_or(depth_mask, segmentation_mask)

            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

        elif strategy == "depth_guided":
            # Use depth to refine segmentation edges (balanced)
            # Strategy:
            # 1. Start with segmentation mask (good at semantic separation)
            # 2. Use depth to refine uncertain regions (edges)

            # Find uncertain region (edges of segmentation)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            seg_eroded = cv2.erode(segmentation_mask, kernel, iterations=2)
            seg_dilated = cv2.dilate(segmentation_mask, kernel, iterations=2)
            uncertain_region = cv2.subtract(seg_dilated, seg_eroded)

            # In uncertain region, trust depth
            # In certain region, trust segmentation
            combined = np.where(
                uncertain_region > 0,
                depth_mask,  # Use depth in uncertain areas
                segmentation_mask  # Use segmentation in certain areas
            ).astype(np.uint8)

            # Final smoothing
            combined = cv2.GaussianBlur(combined, (5, 5), 0)

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: 'intersection', 'union', 'depth_guided'"
            )

        logger.debug(f"Hybrid mask created, foreground pixels: {(combined > 0).sum()}/{combined.size}")

        return combined

    def apply_mask_to_image(
        self,
        image: Image.Image,
        mask: np.ndarray
    ) -> Image.Image:
        """
        Apply mask to image to create RGBA with transparency.

        Args:
            image: Input PIL image
            mask: Binary mask (H, W) with values 0-255

        Returns:
            RGBA PIL image
        """
        # Convert image to RGBA
        image_rgba = np.array(image.convert('RGBA'))

        # Apply mask as alpha channel
        image_rgba[:, :, 3] = mask

        return Image.fromarray(image_rgba)

    def debug_depth_map(
        self,
        image: Image.Image,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> Image.Image:
        """
        Generate colorized depth map for visualization/debugging.

        Args:
            image: Input PIL image
            colormap: OpenCV colormap (COLORMAP_INFERNO, COLORMAP_VIRIDIS, etc.)

        Returns:
            Colorized depth map as PIL image
        """
        depth = self.estimate_depth(image)

        # Convert to 0-255 range
        depth_vis = (depth * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_vis, colormap)

        # Convert BGR to RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return Image.fromarray(depth_colored)


# Convenience function for quick depth-based background removal
def depth_remove_background(
    image: Image.Image,
    device: torch.device,
    model_size: str = 'vitb',
    depth_percentile: int = 30
) -> Image.Image:
    """
    Quick depth-based background removal.

    Args:
        image: Input PIL image
        device: PyTorch device
        model_size: 'vits', 'vitb', or 'vitl'
        depth_percentile: Foreground depth threshold (20-50)

    Returns:
        RGBA image with background removed
    """
    masker = DepthGuidedMasker(device, model_size=model_size)
    mask = masker.create_depth_mask(image, threshold_percentile=depth_percentile)
    return masker.apply_mask_to_image(image, mask)

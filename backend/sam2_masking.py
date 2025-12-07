"""
SAM2-Based Mask Refinement Module

This module uses Meta's Segment Anything Model 2 (SAM2) to refine segmentation masks
for superior edge quality. It's designed to work with the hybrid background removal
pipeline, using automatic bounding box prompts derived from initial segmentation masks.

Key advantages:
- Precise edge detection (hair, fur, fine details)
- Zero-shot segmentation (works on never-seen-before objects)
- Handles transparent/translucent objects well
- Separates objects with similar colors to background
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

# SAM2 will be imported dynamically to avoid errors if not installed
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    logger.warning("SAM2 not installed. Edge refinement will be unavailable.")
    SAM2_AVAILABLE = False


class SAM2Refiner:
    """
    SAM2-based mask refinement using automatic prompts from initial segmentation.

    Uses bounding box prompts derived from rembg/preprocessing masks to guide
    SAM2 for superior edge quality, especially for:
    - Fine details (hair, fur, feathers)
    - Complex edges (transparent objects, reflections)
    - Objects with similar colors to background
    """

    def __init__(self, device: torch.device, model_size: str = 'tiny'):
        """
        Initialize SAM2 model.

        Args:
            device: PyTorch device (cuda, mps, cpu)
            model_size: Model size variant
                - 'tiny': 38MB, ~0.5-1s per image (fast) [RECOMMENDED for speed]
                - 'small': 131MB, ~1-2s per image (balanced)
                - 'base_plus': 224MB, ~2-3s per image (high quality)
                - 'large': 224MB, ~3-4s per image (best quality)

        Raises:
            ImportError: If SAM2 is not installed
            RuntimeError: If model loading fails
        """
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. "
                "Install with: pip install sam2"
            )

        self.device = device
        self.model_size = model_size

        logger.info(f"Initializing SAM2 ({model_size})...")

        # IMPORTANT: SAM2 has CUDA-specific code that doesn't work on MPS
        # For Apple Silicon, we must use CPU
        if device.type == "mps":
            logger.warning(f"SAM2 doesn't support MPS yet, using CPU for SAM2 (tiny model is still fast on CPU)")
            sam2_device = torch.device("cpu")
        else:
            sam2_device = device

        try:
            # Set environment to prevent CUDA compilation attempts
            import os
            old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA from SAM2

            # Load from Hugging Face Hub (auto-downloads to ~/.cache/huggingface/)
            self.predictor = SAM2ImagePredictor.from_pretrained(
                f"facebook/sam2.1-hiera-{model_size}",
                device=str(sam2_device)  # Specify device during loading
            )

            # Restore CUDA visibility
            if old_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)

            self.device = sam2_device  # Store actual device being used
            logger.info(f"SAM2 ({model_size}) loaded successfully on {sam2_device}")

        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")
            raise RuntimeError(f"SAM2 initialization failed: {e}")

    def mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to bounding box [x1, y1, x2, y2].

        Args:
            mask: Binary mask (H, W) with values 0-255

        Returns:
            Bounding box as [x1, y1, x2, y2] numpy array
        """
        # Find all foreground pixels
        coords = np.column_stack(np.where(mask > 0))

        if len(coords) == 0:
            # No foreground found, return full image bbox
            logger.warning("No foreground found in mask, using full image bbox")
            return np.array([0, 0, mask.shape[1], mask.shape[0]])

        # Get min/max coordinates
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)

        # Add small padding (5% of bbox size)
        padding_x = int((x2 - x1) * 0.05)
        padding_y = int((y2 - y1) * 0.05)

        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(mask.shape[1], x2 + padding_x)
        y2 = min(mask.shape[0], y2 + padding_y)

        return np.array([x1, y1, x2, y2])

    @torch.no_grad()
    def refine_mask(
        self,
        image: Image.Image,
        initial_mask: np.ndarray
    ) -> np.ndarray:
        """
        Refine mask using SAM2 with bounding box prompt from initial mask.

        SAM2 excels at:
        - Precise edge detection (hair, fur, fine details)
        - Handling transparent/translucent objects
        - Separating objects with similar colors to background

        Args:
            image: RGB PIL image
            initial_mask: Binary mask (H, W) with values 0-255 from rembg/preprocessing

        Returns:
            Refined binary mask (H, W) with values 0-255
        """
        logger.debug("Refining mask with SAM2...")

        # Convert image to RGB numpy
        image_rgb = np.array(image.convert('RGB'))

        # Get bounding box from initial mask
        bbox = self.mask_to_bbox(initial_mask)

        logger.debug(f"SAM2 bbox prompt: {bbox}")

        # Set image for SAM2 (computes image embeddings)
        self.predictor.set_image(image_rgb)

        # Predict with bounding box prompt
        masks, scores, _ = self.predictor.predict(
            box=bbox,
            multimask_output=False  # Single mask (highest quality)
        )

        # Convert to 0-255 binary mask
        refined_mask = (masks[0] * 255).astype(np.uint8)

        logger.debug(f"SAM2 refinement complete (confidence: {scores[0]:.3f})")

        return refined_mask

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


# Convenience function for quick SAM2-based background removal
def sam2_remove_background(
    image: Image.Image,
    device: torch.device,
    model_size: str = 'tiny',
    initial_mask: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Quick SAM2-based background removal.

    If no initial mask is provided, uses simple thresholding to get initial bbox.

    Args:
        image: Input PIL image
        device: PyTorch device
        model_size: 'tiny', 'small', 'base_plus', or 'large'
        initial_mask: Optional initial mask (if None, auto-generates)

    Returns:
        RGBA image with background removed
    """
    refiner = SAM2Refiner(device, model_size=model_size)

    if initial_mask is None:
        # Simple auto-detection: assume object is in center 80% of image
        h, w = np.array(image).shape[:2]
        padding_h = int(h * 0.1)
        padding_w = int(w * 0.1)
        initial_mask = np.zeros((h, w), dtype=np.uint8)
        initial_mask[padding_h:h-padding_h, padding_w:w-padding_w] = 255

    mask = refiner.refine_mask(image, initial_mask)
    return refiner.apply_mask_to_image(image, mask)

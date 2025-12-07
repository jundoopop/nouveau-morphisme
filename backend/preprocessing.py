"""
Multi-Stage Background Removal Preprocessing Module

This module provides advanced background removal techniques using computer vision
to improve mask quality before feeding images to TripoSR for 3D mesh generation.

Techniques used:
1. Enhanced rembg with alpha matting
2. Morphological operations (noise removal, hole filling)
3. Optional GrabCut refinement (color-based segmentation)
4. Edge smoothing for natural transitions
5. Connected component analysis to keep only main object
"""

import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Initialize rembg session with better model
# Options: "u2net", "u2netp", "u2net_human_seg", "isnet-general-use", "isnet-anime"
REMBG_SESSION = new_session("isnet-general-use")  # Better edge quality than default u2net


def detect_transparency_quality(image: Image.Image) -> dict:
    """
    Analyze if image already has good transparency.

    Detection criteria:
    1. Image has alpha channel (RGBA)
    2. Alpha distribution is binary (mostly 0 or 255)
    3. Foreground is well-defined (contiguous non-zero alpha)
    4. No semi-transparent noise

    Args:
        image: Input PIL image

    Returns:
        Dictionary with quality metrics:
        - has_alpha: bool - Image has alpha channel
        - is_binary: bool - Alpha is mostly 0 or 255
        - binary_ratio: float - Fraction of pixels that are 0 or 255
        - foreground_ratio: float - Fraction of pixels with alpha > 50
        - largest_component_ratio: float - Size of largest component vs total foreground
        - quality_score: float - Overall quality (0-1, higher = better)
        - needs_processing: bool - Whether processing is needed
    """
    # Check if image has alpha
    if image.mode != 'RGBA':
        return {
            "has_alpha": False,
            "is_binary": False,
            "binary_ratio": 0.0,
            "foreground_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "quality_score": 0.0,
            "needs_processing": True
        }

    # Extract alpha channel
    alpha = np.array(image.split()[-1])
    total_pixels = alpha.size

    # Metric 1: Binary distribution
    # Count pixels near 0 (background) and near 255 (foreground)
    near_zero = (alpha <= 10).sum()
    near_full = (alpha >= 245).sum()
    binary_pixels = near_zero + near_full
    binary_ratio = binary_pixels / total_pixels

    # Metric 2: Foreground ratio
    foreground_pixels = (alpha > 50).sum()
    foreground_ratio = foreground_pixels / total_pixels

    # Metric 3: Foreground contiguity (check if foreground is one connected component)
    alpha_binary = (alpha > 128).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        alpha_binary, connectivity=8
    )

    if num_labels <= 1:
        # No foreground detected
        largest_component_ratio = 0.0
    else:
        # Get sizes of all components (excluding background at index 0)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_size = sizes.max() if len(sizes) > 0 else 0
        largest_component_ratio = largest_size / max(foreground_pixels, 1)

    # Calculate quality score (weighted combination)
    is_binary = binary_ratio > 0.85
    is_contiguous = largest_component_ratio > 0.90
    has_reasonable_foreground = 0.05 < foreground_ratio < 0.95

    quality_score = (
        binary_ratio * 0.4 +              # 40% weight on binary alpha
        largest_component_ratio * 0.3 +   # 30% weight on contiguity
        (1.0 if has_reasonable_foreground else 0.5) * 0.3  # 30% weight on foreground size
    )

    needs_processing = not (is_binary and is_contiguous and has_reasonable_foreground)

    result = {
        "has_alpha": True,
        "is_binary": is_binary,
        "binary_ratio": float(binary_ratio),
        "foreground_ratio": float(foreground_ratio),
        "largest_component_ratio": float(largest_component_ratio),
        "quality_score": float(quality_score),
        "needs_processing": needs_processing
    }

    logger.info(
        f"Transparency quality: score={quality_score:.2f}, "
        f"binary={binary_ratio:.2f}, contiguous={largest_component_ratio:.2f}, "
        f"needs_processing={needs_processing}"
    )

    return result


def quantize_alpha(
    mask: np.ndarray,
    threshold: int = 200,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Convert fuzzy alpha to binary (0 or 255).

    This prevents semi-transparent pixels from being interpreted as
    geometry by TripoSR's marching cubes algorithm, which is the root
    cause of mesh flattening.

    Args:
        mask: Alpha channel (H, W) with values 0-255
        threshold: alpha > threshold → 255, else → 0
        fill_holes: Apply morphological closing before quantization

    Returns:
        Binary mask with only 0 or 255 values
    """
    # Fill small holes first to avoid creating gaps during quantization
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Quantize to binary
    quantized = np.where(mask > threshold, 255, 0).astype(np.uint8)

    # Log statistics for debugging
    fuzzy_pixels = ((mask > 0) & (mask < 255)).sum()
    if fuzzy_pixels > 0:
        logger.debug(
            f"Alpha quantization: removed {fuzzy_pixels} fuzzy pixels "
            f"({fuzzy_pixels / mask.size * 100:.1f}%)"
        )

    return quantized


def enhanced_background_removal(
    image: Image.Image,
    processing_mode: str = "auto",
    use_grabcut: bool = True,
    remove_small_components: bool = True,
    kernel_size: int = 5,
    blur_size: int = 5,
    morph_iterations: int = 2
) -> Image.Image:
    """
    Multi-stage background removal pipeline with mode selection.

    Processing Modes:
    - "auto": Detect transparency quality and choose best mode
    - "conservative": Preserve fine details (current behavior with fuzzy edges)
    - "aggressive": Binary alpha quantization, removes stubborn backgrounds

    Args:
        image: Input PIL image
        processing_mode: Processing strategy ("auto", "conservative", "aggressive")
        use_grabcut: Whether to use GrabCut refinement (ignored in aggressive mode)
        remove_small_components: Remove disconnected noise, keep only largest object
        kernel_size: Size of morphological kernel (3-7, larger = more aggressive)
        blur_size: Gaussian blur size for edge smoothing (ignored in aggressive mode)
        morph_iterations: Number of morphological operation iterations (1-3)

    Returns:
        RGBA PIL image with refined alpha mask
    """
    logger.info(f"Starting background removal with mode='{processing_mode}'")

    # ========== AUTO MODE: DETECT AND CHOOSE ==========
    if processing_mode == "auto":
        quality = detect_transparency_quality(image)

        if not quality["needs_processing"]:
            logger.info(
                f"✅ Pre-existing transparency detected "
                f"(quality={quality['quality_score']:.2f}). Skipping processing."
            )
            return image.convert('RGBA')

        # Choose mode based on quality
        if quality["quality_score"] < 0.5:
            processing_mode = "aggressive"
            logger.info(f"Auto-selected aggressive mode (quality={quality['quality_score']:.2f})")
        else:
            processing_mode = "conservative"
            logger.info(f"Auto-selected conservative mode (quality={quality['quality_score']:.2f})")

    # ========== MODE-SPECIFIC PARAMETERS ==========
    if processing_mode == "aggressive":
        alpha_matting_foreground_threshold = 250
        alpha_matting_background_threshold = 5
        morph_kernel_size = 3
        morph_iterations_actual = 4
        use_grabcut_actual = False
        quantize_final = True
        blur_size_actual = 0
    elif processing_mode == "conservative":
        alpha_matting_foreground_threshold = 240
        alpha_matting_background_threshold = 10
        morph_kernel_size = kernel_size
        morph_iterations_actual = morph_iterations
        use_grabcut_actual = use_grabcut
        quantize_final = False
        blur_size_actual = blur_size
    else:
        raise ValueError(
            f"Unknown processing_mode: '{processing_mode}'. "
            f"Choose: 'auto', 'conservative', or 'aggressive'"
        )

    # Stage 1: Initial mask from rembg with alpha matting
    logger.debug("Stage 1: Initial rembg segmentation")
    initial_result = remove(
        image,
        session=REMBG_SESSION,
        alpha_matting=True,                      # Better edge quality
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=10              # Edge refinement size
    )

    # Convert to RGBA if not already
    if initial_result.mode != 'RGBA':
        initial_result = initial_result.convert('RGBA')

    # Extract alpha channel
    alpha = np.array(initial_result.split()[-1])
    logger.debug(f"Initial alpha shape: {alpha.shape}, dtype: {alpha.dtype}")

    # Stage 2: Morphological cleanup
    logger.debug(f"Stage 2: Morphological cleanup (kernel={morph_kernel_size}, iter={morph_iterations_actual})")
    alpha = morphological_cleanup(alpha, kernel_size=morph_kernel_size, iterations=morph_iterations_actual)

    # Stage 3: Optional GrabCut refinement
    if use_grabcut_actual:
        logger.debug("Stage 3: GrabCut refinement")
        image_rgb = np.array(image.convert('RGB'))
        alpha = grabcut_refinement(image_rgb, alpha)
    else:
        logger.debug("Stage 3: Skipped GrabCut (aggressive mode)")

    # Stage 4: Remove small disconnected components
    if remove_small_components:
        logger.debug("Stage 4: Removing small components")
        alpha = keep_largest_component(alpha)

    # Stage 5: Edge processing (mode-dependent)
    if quantize_final:
        logger.debug("Stage 5: Alpha quantization (binary)")
        alpha = quantize_alpha(alpha, threshold=200, fill_holes=True)
    else:
        logger.debug("Stage 5: Edge smoothing (fuzzy)")
        alpha = smooth_edges(alpha, blur_size=blur_size_actual)

    # Stage 6: Reconstruct RGBA image
    result_rgba = np.array(image.convert('RGBA'))
    result_rgba[:, :, 3] = alpha

    logger.info("Multi-stage background removal complete")
    return Image.fromarray(result_rgba)


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    """
    Remove noise and fill small holes using morphological operations.

    Args:
        mask: Binary mask (0-255)
        kernel_size: Size of structuring element
        iterations: Number of iterations for morphological operations (1-3)

    Returns:
        Cleaned mask
    """
    # Create elliptical kernel (better for organic shapes than rectangular)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Remove small noise: opening = erosion → dilation
    # This removes small white spots in the background
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Fill small holes: closing = dilation → erosion
    # This fills small black holes in the foreground
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return mask


def grabcut_refinement(
    image_rgb: np.ndarray,
    initial_mask: np.ndarray,
    iterations: int = 5
) -> np.ndarray:
    """
    Use GrabCut to refine mask using color information.

    GrabCut is particularly good at:
    - Objects with similar colors to background (where depth helps)
    - Refining edges using color consistency
    - Handling gradual transitions

    Args:
        image_rgb: RGB image (H, W, 3)
        initial_mask: Binary mask from previous stages (H, W)
        iterations: Number of GrabCut iterations (3-10)

    Returns:
        Refined binary mask
    """
    # Convert binary mask to GrabCut format
    # GC_BGD (0) = definite background
    # GC_FGD (1) = definite foreground
    # GC_PR_BGD (2) = probably background
    # GC_PR_FGD (3) = probably foreground

    # We use a conservative approach:
    # - High alpha values (>200) = definite foreground
    # - Low alpha values (<50) = definite background
    # - Middle values = let GrabCut decide

    grabcut_mask = np.where(initial_mask > 200, cv2.GC_FGD,
                   np.where(initial_mask < 50, cv2.GC_BGD, cv2.GC_PR_FGD)).astype('uint8')

    # Initialize models (required by GrabCut algorithm)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        # Run GrabCut
        cv2.grabCut(
            image_rgb,
            grabcut_mask,
            None,  # No bounding box (using mask instead)
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_MASK
        )

        # Convert back to binary mask
        # Keep pixels marked as foreground or probably foreground
        refined_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype('uint8')

        return refined_mask

    except cv2.error as e:
        logger.warning(f"GrabCut failed: {e}. Returning initial mask.")
        return initial_mask


def keep_largest_component(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Remove small disconnected components, keep only the largest (main object).

    This is useful to remove:
    - Background noise that passed through initial segmentation
    - Small artifacts from alpha matting
    - Disconnected shadow regions

    Args:
        mask: Binary mask
        min_size: Minimum component size to consider (in pixels)

    Returns:
        Mask with only largest component
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        # No foreground components found
        return mask

    # Get sizes of all components (excluding background at index 0)
    sizes = stats[1:, cv2.CC_STAT_AREA]

    if len(sizes) == 0:
        return mask

    # Keep only largest component
    largest_label = 1 + np.argmax(sizes)

    # Create clean mask with only largest component
    clean_mask = np.where(labels == largest_label, 255, 0).astype('uint8')

    logger.debug(f"Removed {num_labels - 2} small components, kept largest ({sizes.max()} pixels)")

    return clean_mask


def smooth_edges(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """
    Smooth mask edges to reduce aliasing artifacts.

    Uses Gaussian blur for natural edge falloff, which:
    - Reduces jagged edges
    - Creates natural alpha transitions
    - Prevents harsh cutoffs that cause artifacts in 3D

    Args:
        mask: Binary or grayscale mask
        blur_size: Gaussian kernel size (must be odd, 3-11)

    Returns:
        Smoothed mask
    """
    # Ensure blur_size is odd
    if blur_size % 2 == 0:
        blur_size += 1

    # Apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def get_mask_only(
    image: Image.Image,
    use_grabcut: bool = True,
    kernel_size: int = 5,
    blur_size: int = 5,
    morph_iterations: int = 2
) -> np.ndarray:
    """
    Get only the alpha mask without applying it to the image.
    Useful for debugging or combining with depth masks.

    Args:
        image: Input PIL image
        use_grabcut: Whether to use GrabCut refinement
        kernel_size: Size of morphological kernel (3-7)
        blur_size: Gaussian blur size for edge smoothing (3-7)
        morph_iterations: Number of morphological operation iterations (1-3)

    Returns:
        Binary mask as numpy array (H, W) with values 0-255
    """
    result = enhanced_background_removal(
        image,
        use_grabcut=use_grabcut,
        kernel_size=kernel_size,
        blur_size=blur_size,
        morph_iterations=morph_iterations
    )
    return np.array(result.split()[-1])


# Convenience function for quick testing
def quick_remove_bg(image: Image.Image, quality: str = "balanced") -> Image.Image:
    """
    Quick background removal with preset quality levels.

    Args:
        image: Input PIL image
        quality: "fast", "balanced", or "best"
            - fast: No GrabCut, larger kernels (0.5-1s)
            - balanced: GrabCut enabled, default settings (1-2s)
            - best: GrabCut + smaller kernels for detail (2-3s)

    Returns:
        RGBA image with background removed
    """
    if quality == "fast":
        return enhanced_background_removal(
            image,
            use_grabcut=False,
            kernel_size=7,
            blur_size=7
        )
    elif quality == "balanced":
        return enhanced_background_removal(
            image,
            use_grabcut=True,
            kernel_size=5,
            blur_size=5
        )
    elif quality == "best":
        return enhanced_background_removal(
            image,
            use_grabcut=True,
            kernel_size=3,
            blur_size=3
        )
    else:
        raise ValueError(f"Unknown quality setting: {quality}. Use 'fast', 'balanced', or 'best'")

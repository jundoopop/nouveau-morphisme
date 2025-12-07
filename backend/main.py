import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn
import logging
import json
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Icon Composer Backend")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_device():
    """
    Detects the best available hardware device for PyTorch.
    Priority: CUDA -> MPS (Metal) -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
logger.info(f"🚀 Running on device: {DEVICE}")


@app.get("/")
async def root():
    return {"message": "3D Icon Composer Backend is running", "device": str(DEVICE)}


from rembg import remove
from PIL import Image
import io
from fastapi.responses import Response
from preprocessing import quick_remove_bg, get_mask_only


@app.post("/remove-bg")
async def remove_image_background(file: UploadFile = File(...)):
    """
    Fast background removal endpoint for preview/testing.
    Uses enhanced rembg without depth for speed.
    """
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))

    # Use balanced quality preset (GrabCut enabled, better edges)
    output_image = quick_remove_bg(input_image, quality="best")

    # Save to buffer
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    return Response(content=img_bytes, media_type="image/png")


# Load TripoSR Model
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
import trimesh
import numpy as np

# Initialize TripoSR Model
# This will download the model weights automatically
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.to(DEVICE)

# Initialize Depth-Guided Masker (for hybrid background removal)
from depth_masking import DepthGuidedMasker
from preprocessing import enhanced_background_removal
from sam2_masking import SAM2Refiner

try:
    depth_masker = DepthGuidedMasker(DEVICE, model_size="vitb")
    logger.info("✅ Depth-guided masking enabled (hybrid mode)")
    DEPTH_ENABLED = True
except (ImportError, FileNotFoundError) as e:
    logger.warning(f"⚠️  Depth-guided masking unavailable: {e}")
    logger.warning("   Falling back to multi-stage preprocessing only")
    DEPTH_ENABLED = False

# Initialize SAM2 Refiner (for edge refinement)
SAM2_ENABLED = False
try:
    if DEPTH_ENABLED:  # Only use SAM2 if we have GPU/MPS acceleration
        sam2_refiner = SAM2Refiner(DEVICE, model_size="tiny")
        logger.info("✅ SAM2 edge refinement enabled")
        SAM2_ENABLED = True
except Exception as e:
    logger.warning(f"⚠️  SAM2 edge refinement unavailable: {e}")
    logger.warning("   Using segmentation masks without SAM2 refinement")


class MeshGenerationParams(BaseModel):
    """
    Parameters for mesh generation pipeline.
    All parameters are optional with sensible defaults.
    """

    # Depth masking parameters
    depth_percentile: int = Field(
        default=60,
        ge=20,
        le=90,
        description="Depth percentile threshold (20-90). Higher values include more distant objects.",
    )

    # Morphological operation parameters
    morph_kernel_size: int = Field(
        default=3,
        ge=3,
        le=7,
        description="Morphological kernel size (3-7). Smaller preserves details, larger removes noise.",
    )
    morph_iterations: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Morphological operation iterations (1-3). More iterations = more aggressive cleanup.",
    )

    # Foreground resize parameters
    foreground_ratio: float = Field(
        default=0.95,
        ge=0.7,
        le=1.0,
        description="Foreground resize ratio (0.7-1.0). How much of the canvas the object occupies.",
    )

    # Mesh extraction parameters
    mesh_resolution: int = Field(
        default=256,
        description="Marching cubes resolution (128/256/512). Higher = more detail but slower.",
    )
    marching_cubes_threshold: float = Field(
        default=16.0,
        ge=10.0,
        le=30.0,
        description="Marching cubes density threshold (10-30). Lower = more detail, higher = cleaner.",
    )

    # Debug and SAM2 options
    debug_mode: bool = Field(
        default=False, description="Enable debug mode to save intermediate artifacts."
    )
    save_intermediate: bool = Field(
        default=False, description="Save intermediate processing steps."
    )
    use_sam2: bool = Field(
        default=True, description="Use SAM2 edge refinement if available."
    )


def get_model_config(device_type):
    """
    Returns model configuration based on the device.
    """
    if device_type.type == "cuda":
        return {"resolution": 256}  # High quality for GPU
    elif device_type.type == "mps":
        return {"resolution": 256}  # High quality for Metal (Mac)
    else:
        return {"resolution": 128}  # Lower quality for CPU to save time


def setup_debug_dir(debug_mode: bool):
    """
    Create debug output directory if debug mode is enabled.

    Args:
        debug_mode: Whether debug mode is enabled

    Returns:
        Path object pointing to debug directory, or None if debug mode is disabled
    """
    if not debug_mode:
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path("backend/debug_outputs") / timestamp
    debug_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Debug mode enabled. Output directory: {debug_dir}")
    return debug_dir


def save_debug_artifacts(debug_dir, step_name: str, data):
    """
    Save debug artifacts (images, masks, etc.) to debug directory.

    Args:
        debug_dir: Path to debug directory (or None to skip saving)
        step_name: Name of the processing step
        data: numpy array or PIL Image to save
    """
    if debug_dir is None:
        return

    try:
        from PIL import Image as PILImage

        if isinstance(data, np.ndarray):
            PILImage.fromarray(data).save(debug_dir / f"{step_name}.png")
        elif isinstance(data, PILImage.Image):
            data.save(debug_dir / f"{step_name}.png")
        logger.debug(f"Saved debug artifact: {step_name}.png")
    except Exception as e:
        logger.warning(f"Failed to save debug artifact {step_name}: {e}")


def calculate_mesh_metrics(mesh):
    """
    Calculate quality metrics for a mesh.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        Dictionary of metrics
    """
    try:
        return {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "surface_area": float(mesh.area),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "is_watertight": bool(mesh.is_watertight),
            "bounds": mesh.bounds.tolist(),
            "euler_characteristic": (
                int(mesh.euler_number) if hasattr(mesh, "euler_number") else None
            ),
        }
    except Exception as e:
        logger.warning(f"Failed to calculate some mesh metrics: {e}")
        return {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "error": str(e),
        }


def normalize_mesh(mesh, target_extent: float = 1.5):
    """
    Center and scale mesh so controls orbit around its middle.

    Args:
        mesh: trimesh.Trimesh object
        target_extent: Desired largest dimension after scaling
    """
    try:
        # Clean obvious geometry issues first
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()

        # Recentering keeps orbit/arrow rotations stable
        centroid = mesh.centroid
        mesh.apply_translation(-centroid)

        # Normalize size to a predictable range
        max_extent = float(np.max(mesh.extents))
        if max_extent > 0 and np.isfinite(max_extent):
            scale = target_extent / max_extent
            mesh.apply_scale(scale)

        return mesh
    except Exception as e:
        logger.warning(f"Mesh normalization skipped: {e}")
        return mesh


def save_metrics(debug_dir, metrics: dict, params_used: dict):
    """
    Save mesh quality metrics and parameters to JSON file.

    Args:
        debug_dir: Path to debug directory (or None to skip saving)
        metrics: Dictionary of mesh quality metrics
        params_used: Dictionary of parameters used for generation
    """
    if debug_dir is None:
        return

    try:
        metrics_data = {
            "metrics": metrics,
            "parameters": params_used,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(debug_dir / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Saved metrics to {debug_dir / 'metrics.json'}")
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")


@app.post("/generate-mesh")
async def generate_mesh(file: UploadFile = File(...), params: str = Form(default="{}")):
    """
    Generate 3D mesh from 2D image using hybrid background removal pipeline.

    Pipeline:
    1. Multi-stage preprocessing (rembg + morphological + GrabCut)
    2. Optional: Depth-guided mask refinement (if depth model available)
    3. Hybrid mask combination for best quality
    4. TripoSR 3D mesh generation

    Args:
        file: Uploaded image file
        params: JSON string of MeshGenerationParams (optional, uses defaults if not provided)
    """
    # Parse parameters
    try:
        param_dict = json.loads(params) if params and params != "{}" else {}
        mesh_params = MeshGenerationParams(**param_dict)
        logger.info(f"Mesh generation params: {mesh_params.dict()}")
    except Exception as e:
        logger.error(f"Failed to parse parameters: {e}")
        mesh_params = MeshGenerationParams()  # Use defaults

    # Setup debug directory
    debug_dir = setup_debug_dir(mesh_params.debug_mode or mesh_params.save_intermediate)

    contents = await file.read()
    input_image: Image.Image = Image.open(io.BytesIO(contents))

    logger.info("Starting hybrid background removal pipeline")

    # Ensure input is RGBA for background removal but we might need RGB for models
    if input_image.mode != "RGBA":
        input_image = input_image.convert("RGBA")

    image_rgb = input_image.convert("RGB")  # For depth/TripoSR models

    if DEPTH_ENABLED:
        # ========== HYBRID + SAM2 APPROACH (Best Quality) ==========
        logger.info("Using SAM2 + hybrid depth + segmentation pipeline")

        # Step 1: Get initial segmentation mask from multi-stage preprocessing
        logger.info(
            f"Step 1/4: Multi-stage preprocessing (kernel={mesh_params.morph_kernel_size}, iter={mesh_params.morph_iterations})"
        )
        segmentation_mask = get_mask_only(
            input_image,
            use_grabcut=True,
            kernel_size=mesh_params.morph_kernel_size,
            blur_size=5,
            morph_iterations=mesh_params.morph_iterations,
        )

        # Save segmentation mask for debugging
        save_debug_artifacts(debug_dir, "01_segmentation_mask", segmentation_mask)

        # Step 2: Refine with SAM2 (if enabled)
        if SAM2_ENABLED and mesh_params.use_sam2:
            try:
                logger.info("Step 2/4: SAM2 edge refinement")
                segmentation_mask = sam2_refiner.refine_mask(
                    input_image, segmentation_mask
                )
                save_debug_artifacts(
                    debug_dir, "02_sam2_refined_mask", segmentation_mask
                )
                logger.info("SAM2 refinement complete")
            except Exception as e:
                logger.warning(
                    f"SAM2 refinement failed: {e}. Continuing with unrefined mask."
                )
                # Pipeline continues gracefully with the unrefined mask
        else:
            logger.info("Step 2/4: Skipped (SAM2 not available or disabled)")

        # Step 3: Combine with depth for superior results
        logger.info(
            f"Step 3/4: Depth-guided hybrid masking (percentile={mesh_params.depth_percentile})"
        )
        hybrid_mask = depth_masker.hybrid_mask(
            input_image,
            segmentation_mask,
            depth_percentile=mesh_params.depth_percentile,
            strategy="depth_guided",  # Use depth to refine edges
        )
        save_debug_artifacts(debug_dir, "03_hybrid_mask", hybrid_mask)

        # Step 4: Apply combined mask
        logger.info("Step 4/4: Applying hybrid mask")
        image_processed = depth_masker.apply_mask_to_image(input_image, hybrid_mask)
        save_debug_artifacts(debug_dir, "04_masked_image", image_processed)

        logger.info("Hybrid mask with SAM2 refinement created successfully")

    else:
        # ========== FALLBACK: Multi-stage only (No depth) ==========
        logger.info("Using multi-stage preprocessing (no depth)")

        # Use balanced quality (GrabCut enabled)
        image_processed = enhanced_background_removal(
            input_image,
            use_grabcut=True,  # Better quality, adds ~1s
            remove_small_components=True,
            kernel_size=mesh_params.morph_kernel_size,
            blur_size=5,
            morph_iterations=mesh_params.morph_iterations,
        )

        logger.info("Multi-stage preprocessing complete")

    # TripoSR preprocessing (resize foreground)
    logger.info(f"Resizing foreground (ratio={mesh_params.foreground_ratio})")
    image_processed = resize_foreground(
        image_processed, ratio=mesh_params.foreground_ratio
    )
    save_debug_artifacts(debug_dir, "05_preprocessed_for_triposr", image_processed)

    # DEBUG: Log image info before TripoSR
    logger.info(
        f"Image before TripoSR: mode={image_processed.mode}, size={image_processed.size}"
    )

    # 2. Run TripoSR with error handling
    logger.info("Generating 3D mesh with TripoSR")
    try:
        with torch.no_grad():
            # Ensure RGB for TripoSR
            if image_processed.mode == "RGBA":
                # Create RGB background (white) to handle transparency
                bg = Image.new("RGB", image_processed.size, (255, 255, 255))
                bg.paste(image_processed, mask=image_processed.split()[3])
                model_input = bg
            else:
                model_input = image_processed.convert("RGB")

            scene_codes = model(model_input, device=DEVICE)
    except Exception as e:
        logger.error(f"TripoSR failed: {type(e).__name__}: {e}")
        logger.error(
            f"Image shape: {np.array(image_processed).shape}, dtype: {np.array(image_processed).dtype}"
        )
        raise

    # 3. Extract Mesh
    logger.info(
        f"Extracting mesh (resolution={mesh_params.mesh_resolution}, "
        f"threshold={mesh_params.marching_cubes_threshold})"
    )

    meshes = model.extract_mesh(
        scene_codes,
        has_vertex_color=True,
        resolution=mesh_params.mesh_resolution,
        threshold=mesh_params.marching_cubes_threshold,
    )
    mesh = meshes[0]

    # Refine mesh surface (Laplacian smoothing)
    logger.info("Refining mesh surface...")
    try:
        trimesh.smoothing.filter_laplacian(mesh, iterations=5)
        logger.info("✅ Mesh smoothing applied")
    except Exception as e:
        logger.warning(f"⚠️ Mesh smoothing failed: {e}")

    # Normalize mesh for consistent viewer controls
    mesh = normalize_mesh(mesh)

    # Calculate mesh quality metrics
    mesh_metrics = calculate_mesh_metrics(mesh)
    logger.info(f"Mesh metrics: {mesh_metrics}")

    # Save metrics and parameters if debug mode enabled
    save_metrics(debug_dir, mesh_metrics, mesh_params.dict())

    # 4. Export to GLB
    logger.info("Exporting mesh to GLB format")
    glb_data = mesh.export(file_type="glb")

    logger.info("✅ Mesh generation complete")
    return Response(content=glb_data, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

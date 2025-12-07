import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

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

    # Use fast quality preset (no GrabCut, optimized for speed)
    output_image = quick_remove_bg(input_image, quality="fast")

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


@app.post("/generate-mesh")
async def generate_mesh(file: UploadFile = File(...)):
    """
    Generate 3D mesh from 2D image using hybrid background removal pipeline.

    Pipeline:
    1. Multi-stage preprocessing (rembg + morphological + GrabCut)
    2. Optional: Depth-guided mask refinement (if depth model available)
    3. Hybrid mask combination for best quality
    4. TripoSR 3D mesh generation
    """
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
        logger.info("Step 1/4: Multi-stage preprocessing")
        segmentation_mask = get_mask_only(input_image, use_grabcut=True)

        # DEBUG: Save rembg mask
        Image.fromarray(segmentation_mask).save("debug_rembg_mask.png")

        # Step 2: Refine with SAM2 (if enabled)
        if SAM2_ENABLED:
            logger.info("Step 2/4: SAM2 edge refinement")
            segmentation_mask = sam2_refiner.refine_mask(input_image, segmentation_mask)
            Image.fromarray(segmentation_mask).save("debug_sam2_mask.png")
            logger.info("SAM2 refinement complete")
        else:
            logger.info("Step 2/4: Skipped (SAM2 not available)")

        # Step 3: Combine with depth for superior results
        logger.info("Step 3/4: Depth-guided hybrid masking")
        hybrid_mask = depth_masker.hybrid_mask(
            input_image,
            segmentation_mask,
            depth_percentile=35,  # Tune this: 30-40 typical, lower=closer objects only
            strategy="depth_guided",  # Use depth to refine edges
        )

        # Step 4: Apply combined mask
        logger.info("Step 4/4: Applying hybrid mask")
        image_processed = depth_masker.apply_mask_to_image(input_image, hybrid_mask)

        logger.info("Hybrid mask with SAM2 refinement created successfully")

    else:
        # ========== FALLBACK: Multi-stage only (No depth) ==========
        logger.info("Using multi-stage preprocessing (no depth)")

        # Use balanced quality (GrabCut enabled)
        image_processed = enhanced_background_removal(
            input_image,
            use_grabcut=True,  # Better quality, adds ~1s
            remove_small_components=True,
            kernel_size=5,
            blur_size=5,
        )

        logger.info("Multi-stage preprocessing complete")

    # TripoSR preprocessing (resize foreground)
    image_processed = resize_foreground(image_processed, ratio=0.85)

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
    config = get_model_config(DEVICE)
    logger.info(f"Extracting mesh with config: {config}")

    meshes = model.extract_mesh(
        scene_codes, has_vertex_color=True, resolution=config["resolution"]
    )
    mesh = meshes[0]

    # Refine mesh surface (Laplacian smoothing)
    logger.info("Refining mesh surface...")
    try:
        trimesh.smoothing.filter_laplacian(mesh, iterations=5)
        logger.info("✅ Mesh smoothing applied")
    except Exception as e:
        logger.warning(f"⚠️ Mesh smoothing failed: {e}")

    # 4. Export to GLB
    logger.info("Exporting mesh to GLB format")
    glb_data = mesh.export(file_type="glb")

    logger.info("✅ Mesh generation complete")
    return Response(content=glb_data, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

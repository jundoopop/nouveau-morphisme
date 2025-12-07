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

try:
    depth_masker = DepthGuidedMasker(DEVICE, model_size='vitb')
    logger.info("✅ Depth-guided masking enabled (hybrid mode)")
    DEPTH_ENABLED = True
except (ImportError, FileNotFoundError) as e:
    logger.warning(f"⚠️  Depth-guided masking unavailable: {e}")
    logger.warning("   Falling back to multi-stage preprocessing only")
    DEPTH_ENABLED = False


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
    input_image = Image.open(io.BytesIO(contents))

    logger.info("Starting hybrid background removal pipeline")

    if DEPTH_ENABLED:
        # ========== HYBRID APPROACH (Best Quality) ==========
        logger.info("Using hybrid depth + segmentation pipeline")

        # Step 1: Get segmentation mask from multi-stage preprocessing
        segmentation_mask = get_mask_only(input_image, use_grabcut=True)

        # Step 2: Combine with depth for superior results
        hybrid_mask = depth_masker.hybrid_mask(
            input_image,
            segmentation_mask,
            depth_percentile=35,  # Tune this: 30-40 typical, lower=closer objects only
            strategy="depth_guided"  # Use depth to refine edges
        )

        # Step 3: Apply combined mask
        image_processed = depth_masker.apply_mask_to_image(input_image, hybrid_mask)

        logger.info("Hybrid mask created successfully")

    else:
        # ========== FALLBACK: Multi-stage only (No depth) ==========
        logger.info("Using multi-stage preprocessing (no depth)")

        # Use balanced quality (GrabCut enabled)
        image_processed = enhanced_background_removal(
            input_image,
            use_grabcut=True,  # Better quality, adds ~1s
            remove_small_components=True,
            kernel_size=5,
            blur_size=5
        )

        logger.info("Multi-stage preprocessing complete")

    # TripoSR preprocessing (resize foreground)
    image_processed = resize_foreground(image_processed, ratio=0.85)

    # DEBUG: Log image info before TripoSR
    logger.info(f"Image before TripoSR: mode={image_processed.mode}, size={image_processed.size}")

    # 2. Run TripoSR with error handling
    logger.info("Generating 3D mesh with TripoSR")
    try:
        with torch.no_grad():
            scene_codes = model(image_processed, device=DEVICE)
    except Exception as e:
        logger.error(f"TripoSR failed: {type(e).__name__}: {e}")
        logger.error(f"Image shape: {np.array(image_processed).shape}, dtype: {np.array(image_processed).dtype}")
        raise

    # 3. Extract Mesh
    config = get_model_config(DEVICE)
    logger.info(f"Extracting mesh with config: {config}")

    meshes = model.extract_mesh(scene_codes, resolution=config["resolution"])
    mesh = meshes[0]

    # 4. Export to GLB
    logger.info("Exporting mesh to GLB format")
    glb_data = mesh.export(file_type="glb")

    logger.info("✅ Mesh generation complete")
    return Response(content=glb_data, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

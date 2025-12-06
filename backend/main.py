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


@app.post("/remove-bg")
async def remove_image_background(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))

    # Remove background
    output_image = remove(input_image)

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
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))

    # 1. Preprocess Image
    # Remove background if not already transparent (TripoSR expects transparent bg)
    # We can use rembg or TripoSR's utility.
    # Let's use TripoSR's utility for consistency with their pipeline.
    # Note: tsr.utils.remove_background might use rembg internally or similar.

    # Ensure RGBA
    if input_image.mode != "RGBA":
        input_image = input_image.convert("RGBA")

    # Remove background
    # We already have rembg imported as remove, but let's use the one from tsr.utils if it's better suited
    # actually tsr.utils.remove_background takes an image and returns an image with bg removed.
    # But let's stick to our previous rembg if we want, OR use tsr.utils.
    # Let's use tsr.utils.remove_background to be safe with what the model expects.

    # However, looking at TripoSR source, they often use rembg.
    # Let's use the imported remove_background from tsr.utils

    # Preprocessing
    image_processed = remove_background(input_image, rembg_session=None)
    image_processed = resize_foreground(image_processed, ratio=0.85)

    # 2. Run TripoSR
    with torch.no_grad():
        scene_codes = model(image_processed, device=DEVICE)

    # 3. Extract Mesh
    config = get_model_config(DEVICE)
    logger.info(f"Generating mesh with config: {config}")

    meshes = model.extract_mesh(scene_codes, resolution=config["resolution"])
    mesh = meshes[0]

    # 4. Export to GLB
    # TripoSR returns a trimesh object
    glb_data = mesh.export(file_type="glb")

    return Response(content=glb_data, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

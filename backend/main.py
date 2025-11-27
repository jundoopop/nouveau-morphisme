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
async def remove_background(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))

    # Remove background
    output_image = remove(input_image)

    # Save to buffer
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    return Response(content=img_bytes, media_type="image/png")


# Load Depth Model
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
import trimesh

# Initialize Depth Model (Lazy load or global)
# Using Depth Anything V2 for better details
depth_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)


@app.post("/generate-mesh")
async def generate_mesh(file: UploadFile = File(...)):
    contents = await file.read()
    original_image = Image.open(io.BytesIO(contents))

    # 0. Remove Background (Ensure RGBA)
    input_image = remove(original_image).convert("RGBA")

    # Create RGB copy for depth estimation (Depth Anything expects 3 channels)
    input_image_rgb = input_image.convert("RGB")

    # 1. Estimate Depth
    inputs = depth_processor(images=input_image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Normalize depth
    depth_map = prediction.squeeze().cpu().numpy()
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    # 2. Create Mesh
    # Downsample for performance if needed, but let's try full res or slightly reduced
    # For an icon, 256x256 is plenty for geometry
    mesh_res = 256
    image_small = input_image.resize((mesh_res, mesh_res))

    # Extract alpha for masking
    alpha_small = np.array(image_small)[:, :, 3] / 255.0

    depth_small = (
        torch.nn.functional.interpolate(
            torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0),
            size=(mesh_res, mesh_res),
            mode="bilinear",
        )
        .squeeze()
        .numpy()
    )

    # Create vertices
    x = np.linspace(-1, 1, mesh_res)
    y = np.linspace(-1, 1, mesh_res)
    xv, yv = np.meshgrid(x, -y)  # Flip Y for 3D coords

    # Depth extrusion amount
    extrusion_scale = 0.5
    zv = depth_small * extrusion_scale

    vertices = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)

    # Create faces with alpha masking
    faces = []
    alpha_threshold = 0.1  # Threshold to consider a pixel "visible"

    for i in range(mesh_res - 1):
        for j in range(mesh_res - 1):
            # Grid indices
            idx = i * mesh_res + j

            # Check alpha of the 4 corners of the quad
            # (i, j), (i, j+1), (i+1, j), (i+1, j+1)
            a1 = alpha_small[i, j]
            a2 = alpha_small[i, j + 1]
            a3 = alpha_small[i + 1, j]
            a4 = alpha_small[i + 1, j + 1]

            # If any vertex is transparent, skip the quad (or use average)
            # Using max > threshold ensures we keep edges, min > threshold might shrink it too much
            if max(a1, a2, a3, a4) > alpha_threshold:
                # Triangle 1
                faces.append([idx, idx + 1, idx + mesh_res])
                # Triangle 2
                faces.append([idx + 1, idx + mesh_res + 1, idx + mesh_res])

    faces_np = np.array(faces)

    # Create UVs
    u = np.linspace(0, 1, mesh_res)
    v = np.linspace(1, 0, mesh_res)  # Flip V for texture
    uv, vv = np.meshgrid(u, v)
    uvs = np.stack([uv.flatten(), vv.flatten()], axis=1)

    # Create Trimesh object
    # Filter vertices that are not used?
    # Trimesh can handle unused vertices, but for cleaner export we might want to process_()

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces_np,
        visual=trimesh.visual.TextureVisuals(uv=uvs, image=image_small),
        process=False,  # Don't auto-process to keep our UVs aligned
    )

    # Export to GLB
    glb_data = mesh.export(file_type="glb")

    return Response(content=glb_data, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# 3D Icon Composer

A full-stack AI-powered application that transforms 2D images into interactive 3D models with dual rendering engines (WebGL + WebGPU).

## Features

### Backend (FastAPI + AI)
- 🤖 **AI-Powered 3D Generation**: Uses TripoSR (Stability AI) to convert 2D images to 3D meshes
- 🎨 **Background Removal**: Automatic background removal using rembg
- ⚡ **Hardware Optimization**: Auto-detects and uses CUDA/Metal/CPU
- 📦 **GLB Export**: Generates industry-standard 3D model files

### Frontend (Next.js + React)
- 🖼️ **Image Upload**: Drag & drop or click to upload images
- 🔄 **Dual Rendering**: Switch between WebGL (Three.js) and WebGPU
- 🎭 **Shader Modes**:
  - **WebGL**: Default, Toon, Shiny, Wireframe, Normal (5 modes)
  - **WebGPU**: PBR with Cook-Torrance BRDF, Normal (2 modes)
- 💡 **Lighting Presets**: City, Sunset, Studio, Night environments
- 🎮 **Interactive Controls**: Orbit camera, zoom, keyboard controls
- ⚙️ **Physically-Based Rendering**: True PBR with energy conservation, Fresnel effects, ACES tonemapping

## Architecture

```
term/
├── backend/          # FastAPI server with AI models
│   ├── main.py       # API endpoints
│   └── requirements.txt
└── frontend/         # Next.js 16 application
    ├── app/          # Next.js App Router
    ├── components/   # React components
    └── lib/          # WebGPU rendering engine
        └── webgpu/
            ├── core/       # Device, context, renderer
            ├── geometry/   # Procedural geometry
            └── shaders/    # WGSL shaders (PBR)
```

## Prerequisites

- **Python**: 3.10+ (tested with 3.12.7)
- **Node.js**: 18+ (tested with 22.16.0)
- **npm**: 8+ (tested with 11.6.2)
- **GPU** (optional): NVIDIA GPU with CUDA or Apple Silicon with Metal for faster processing

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd term
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will download TripoSR model weights (~2GB) automatically.

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Running the Project

You need to run both backend and frontend servers simultaneously.

### Option A: Using Two Terminal Windows

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # If using virtual environment
python main.py
```
Backend will run on: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Frontend will run on: `http://localhost:3000`

### Option B: Using a Process Manager (tmux/screen)

```bash
# Terminal 1
tmux new -s backend
cd backend && python main.py

# Detach: Ctrl+B then D

# Terminal 2
tmux new -s frontend
cd frontend && npm run dev
```

### Option C: Using npm-run-all (if configured)

```bash
# From project root
npm install -g npm-run-all
npm-run-all --parallel backend frontend
```

## Usage

1. **Open the application**: Navigate to `http://localhost:3000`

2. **Upload an image**:
   - Click the upload area or drag & drop an image
   - Supported formats: PNG, JPG, JPEG, WebP

3. **Wait for processing**:
   - Backend removes background (~2-3 seconds)
   - Backend generates 3D mesh (~10-30 seconds depending on GPU)

4. **Interact with 3D model**:
   - **Mouse**: Drag to rotate, scroll to zoom
   - **Keyboard**: Arrow keys for manual rotation
   - **Shader**: Select from dropdown (Default, Toon, Shiny, etc.)
   - **Lighting**: Choose environment preset
   - **Renderer**: Toggle between WebGL and WebGPU

## API Endpoints

### Backend (Port 8000)

| Endpoint | Method | Description | Input | Output |
|----------|--------|-------------|-------|--------|
| `/` | GET | Health check | None | JSON status |
| `/remove-bg` | POST | Remove background | Image file | PNG with alpha |
| `/generate-mesh` | POST | Generate 3D mesh | Image file | GLB file |

Example curl:
```bash
# Remove background
curl -X POST -F "file=@image.jpg" http://localhost:8000/remove-bg -o output.png

# Generate 3D mesh
curl -X POST -F "file=@image.jpg" http://localhost:8000/generate-mesh -o model.glb
```

## Browser Compatibility

### WebGL Mode (Three.js)
- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari

### WebGPU Mode (Experimental)
- ✅ Chrome 113+ (with WebGPU enabled)
- ✅ Edge 113+
- ⚠️ Firefox (behind flag)
- ❌ Safari (not yet supported)

**Enable WebGPU in Chrome/Edge:**
- Visit `chrome://flags/#enable-unsafe-webgpu`
- Set to "Enabled"
- Restart browser

## Hardware Performance

### GPU Processing (Recommended)
- **NVIDIA GPU**: CUDA support, fastest (256px resolution)
- **Apple Silicon**: Metal acceleration, fast (256px resolution)
- **AMD GPU**: CPU fallback, slower (128px resolution)

### CPU Processing
- Slower mesh generation (30-60 seconds)
- Lower resolution (128px)
- Still functional for testing

## Development

### Frontend Development

```bash
cd frontend

# Development server with hot reload
npm run dev

# Type checking
npm run type-check

# Build for production
npm run build

# Run production build
npm start
```

### Backend Development

```bash
cd backend

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
term/
├── backend/
│   ├── main.py                 # FastAPI app with 3 endpoints
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page
│   │   └── globals.css         # Global styles
│   │
│   ├── components/
│   │   ├── IconGenerator.tsx  # Main UI component
│   │   ├── Viewer3D.tsx       # WebGL renderer
│   │   └── Viewer3DWebGPU.tsx # WebGPU renderer
│   │
│   └── lib/webgpu/
│       ├── core/
│       │   ├── device.ts       # GPU initialization
│       │   ├── context.ts      # Canvas setup
│       │   └── renderer.ts     # Render pipeline
│       │
│       ├── geometry/
│       │   └── torus-knot-generator.ts
│       │
│       └── shaders/
│           ├── vertex.wgsl.ts
│           ├── fragment-default.wgsl.ts  # PBR shader
│           ├── fragment-normal.wgsl.ts
│           └── common/
│               └── pbr-functions.wgsl.ts # Shared PBR math
│
└── README.md
```

## Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'tsr'`
```bash
# Reinstall TripoSR
pip install git+https://github.com/VAST-AI-Research/TripoSR.git
```

**Problem**: Out of memory during mesh generation
```bash
# Solution: Use CPU or reduce input image size
# The backend automatically adjusts resolution based on device
```

**Problem**: CUDA not detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, verify NVIDIA drivers and CUDA toolkit
```

### Frontend Issues

**Problem**: WebGPU not available
- Check browser compatibility (Chrome 113+)
- Enable WebGPU flag: `chrome://flags/#enable-unsafe-webgpu`
- Restart browser

**Problem**: Port 3000 already in use
```bash
# Use different port
PORT=3001 npm run dev
```

**Problem**: Shader compilation errors
- Open browser DevTools console
- Check for WGSL syntax errors
- Verify GPU drivers are up to date

### CORS Issues

If running on different ports/domains:
```python
# backend/main.py - CORS is already enabled for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Tips

1. **Use GPU**: Significant speedup for mesh generation
2. **Resize images**: Smaller images process faster (recommended: 512x512)
3. **WebGPU**: Better performance than WebGL for complex shaders
4. **Close other apps**: Free up VRAM for model processing

## Technology Stack

### Backend
- **FastAPI**: Web framework
- **TripoSR**: AI 3D generation (Stability AI)
- **rembg**: Background removal
- **PyTorch**: Deep learning framework
- **Trimesh**: 3D mesh manipulation

### Frontend
- **Next.js 16**: React framework
- **React 19**: UI library
- **Three.js**: WebGL rendering
- **React Three Fiber**: React bindings for Three.js
- **WebGPU**: Modern GPU API
- **WGSL**: WebGPU Shading Language
- **gl-matrix**: Matrix operations
- **Tailwind CSS**: Styling

## Known Limitations

1. **WebGPU GLTF Loading**: Not yet implemented (Phase 1 priority)
   - WebGPU mode only shows procedural torus knot
   - AI-generated models only render in WebGL mode

2. **Export Function**: Button exists but not implemented
   - Planned: PNG, ICO, SVG export
   - Planned: Multiple resolutions and angles

3. **Single Light**: WebGPU shader supports only one directional light
   - Planned: Multiple lights, point/spot lights

4. **No Shadows**: Shadow mapping not implemented
   - Planned for advanced rendering phase

## Roadmap

- [ ] **Phase 1**: WebGPU GLTF loading (CRITICAL)
- [ ] **Phase 2**: Texture & material system
- [ ] **Phase 3**: Additional shader modes (Toon, Wireframe)
- [ ] **Phase 4**: Export functionality
- [ ] **Phase 5**: Leva controls integration
- [ ] **Phase 6**: Advanced rendering (shadows, IBL, SSAO)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes using EC conventions:
   ```
   feat(scope): add amazing feature

   detailed description of what changed and why
   ```
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Your License Here]

## Credits

- **TripoSR**: Stability AI & Tripo AI
- **rembg**: Daniel Gatis
- **Three.js**: Mr.doob and contributors
- **WebGPU**: W3C Community Group

## Support

For issues and questions:
- Open an issue on GitHub
- Check troubleshooting section
- Review browser console for errors

---

**Built with Claude Code** 🤖

Last updated: December 2024

# Quick Start Guide

## 🚀 One-Command Start

```bash
./start.sh
```

This will:
1. ✅ Check prerequisites (Python, Node.js, npm)
2. 📦 Install missing dependencies
3. 🔧 Start backend on port 8000
4. 🎨 Start frontend on port 3000
5. 📱 Open http://localhost:3000

---

## 🏃 Manual Start (Two Terminals)

### Terminal 1 - Backend
```bash
cd backend
python3 main.py
```
Backend runs on: **http://localhost:8000**

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```
Frontend runs on: **http://localhost:3000**

---

## ✅ First Time Setup

### 1. Install Python Dependencies
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Node Dependencies
```bash
cd frontend
npm install
```

### 3. Run
```bash
./start.sh
```

---

## 🎮 Using the Application

### Upload Image
1. Open http://localhost:3000
2. Click upload area or drag & drop image
3. Wait for processing (~10-30 seconds)

### 3D Controls
- **Rotate**: Mouse drag
- **Zoom**: Mouse scroll
- **Manual Rotation**: Arrow keys

### Renderer Options
- **WebGL Mode**: Full features (5 shader modes)
- **WebGPU Mode**: Modern PBR rendering (experimental)

### Shader Modes
- **Default**: Standard PBR materials
- **Toon**: Cartoon-style cel shading
- **Shiny**: High metalness & low roughness
- **Wireframe**: Green edge visualization
- **Normal**: Normal map visualization

### Lighting Presets
- **City**: Urban environment lighting
- **Sunset**: Warm golden tones
- **Studio**: Professional neutral lighting
- **Night**: Dark atmospheric lighting

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python3 --version  # Should be 3.10+

# Reinstall dependencies
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json .next
npm install
```

### Port already in use
```bash
# Backend (change port in main.py)
# Frontend
PORT=3001 npm run dev
```

### WebGPU not available
1. Use Chrome 113+ or Edge 113+
2. Enable: `chrome://flags/#enable-unsafe-webgpu`
3. Restart browser

---

## 📊 System Requirements

### Minimum
- CPU: Any modern processor
- RAM: 8GB
- Storage: 5GB free
- GPU: Not required (CPU fallback)

### Recommended
- CPU: Multi-core processor
- RAM: 16GB+
- Storage: 10GB free
- GPU: NVIDIA (CUDA) or Apple Silicon (Metal)

---

## 🔗 Useful URLs

- **Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc

---

## 🆘 Common Issues

### "No module named 'tsr'"
```bash
pip install git+https://github.com/VAST-AI-Research/TripoSR.git
```

### "CUDA not available"
- System will fallback to CPU automatically
- Mesh generation will be slower
- Install NVIDIA drivers for CUDA support

### "Port 3000 in use"
```bash
# Kill process using port
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm run dev
```

### Shader errors in console
- Update GPU drivers
- Try WebGL mode instead of WebGPU
- Check browser compatibility

---

## 📝 Development Commands

### Backend
```bash
cd backend

# Run with auto-reload
uvicorn main:app --reload

# Run on different port
uvicorn main:app --port 8001
```

### Frontend
```bash
cd frontend

# Development
npm run dev

# Type check
npx tsc --noEmit

# Build production
npm run build

# Run production
npm start
```

---

## 🎯 Quick Test

### Test Backend API
```bash
# Health check
curl http://localhost:8000

# Remove background
curl -X POST -F "file=@test.jpg" http://localhost:8000/remove-bg -o output.png

# Generate 3D mesh
curl -X POST -F "file=@test.jpg" http://localhost:8000/generate-mesh -o model.glb
```

### Test Frontend
1. Open http://localhost:3000
2. You should see the upload interface
3. Try uploading an image

---

## ⚡ Performance Tips

1. **Use smaller images** (512x512 recommended)
2. **Use GPU** if available (10x faster)
3. **Close other apps** to free VRAM
4. **WebGPU mode** for better shader performance

---

## 📞 Getting Help

1. Check [README.md](README.md) for detailed docs
2. Review logs:
   ```bash
   tail -f logs/backend.log
   tail -f logs/frontend.log
   ```
3. Open browser DevTools Console for frontend errors
4. Check GitHub issues

---

**Ready to create 3D icons? Run `./start.sh` and open http://localhost:3000!** 🚀

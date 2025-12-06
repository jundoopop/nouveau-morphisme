#!/bin/bash

# 3D Icon Composer - Startup Script
# This script starts both backend and frontend servers

set -e

echo "🚀 Starting 3D Icon Composer..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is not installed. Please install Node.js 18+"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is not installed. Please install npm 8+"
    exit 1
fi

echo "✅ Python $(python3 --version | cut -d' ' -f2)"
echo "✅ Node $(node --version)"
echo "✅ npm $(npm --version)"
echo ""

# Check if backend dependencies are installed
echo "📦 Checking backend dependencies..."
if [ -f "backend/venv/bin/python" ]; then
    echo "✅ Virtual environment found"
    PYTHON_CMD="backend/venv/bin/python"
else
    echo "⚠️  No virtual environment found, using system Python"
    PYTHON_CMD="python3"
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
    echo "✅ Frontend dependencies installed"
else
    echo "✅ Frontend dependencies found"
fi

echo ""
echo "🎯 Starting services..."
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Start backend
echo "${BLUE}🔧 Starting Backend (FastAPI)...${NC}"
cd backend
$PYTHON_CMD main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check logs/backend.log"
    exit 1
fi

echo "${GREEN}✅ Backend running on http://localhost:8000${NC}"
echo ""

# Start frontend
echo "${BLUE}🎨 Starting Frontend (Next.js)...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 5

echo "${GREEN}✅ Frontend running on http://localhost:3000${NC}"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "${GREEN}🎉 3D Icon Composer is ready!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Application: ${BLUE}http://localhost:3000${NC}"
echo "🔌 API Docs:    ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo "💡 Tips:"
echo "  - Upload an image to generate 3D models"
echo "  - Toggle between WebGL and WebGPU renderers"
echo "  - Try different shader modes and lighting presets"
echo ""
echo "📋 Logs:"
echo "  - Backend:  tail -f logs/backend.log"
echo "  - Frontend: tail -f logs/frontend.log"
echo ""
echo "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and show logs
tail -f logs/backend.log logs/frontend.log

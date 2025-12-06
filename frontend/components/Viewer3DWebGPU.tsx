"use client";

import { useEffect, useRef, useState } from "react";
import { initWebGPU } from "../lib/webgpu/core/device";
import { configureContext } from "../lib/webgpu/core/context";
import { WebGPURenderer, ShaderMode } from "../lib/webgpu/core/renderer";
import { generateTorusKnot } from "../lib/webgpu/geometry/torus-knot-generator";

interface Viewer3DWebGPUProps {
  imageUrl?: string;
  meshUrl?: string;
  onSwitchToWebGL?: () => void;
}

export default function Viewer3DWebGPU({ imageUrl, meshUrl, onSwitchToWebGL }: Viewer3DWebGPUProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WebGPURenderer | null>(null);
  const animationFrameRef = useRef<number>(0);

  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shaderMode, setShaderMode] = useState<ShaderMode>("Normal");

  // Camera state
  const cameraRef = useRef({
    distance: 5,
    theta: 0, // horizontal angle
    phi: Math.PI / 3, // vertical angle (60 degrees down from top)
  });

  // Mouse interaction state
  const mouseRef = useRef({
    isDragging: false,
    lastX: 0,
    lastY: 0,
  });

  // Initialize WebGPU
  useEffect(() => {
    const initializeWebGPU = async () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      try {
        // Initialize WebGPU device
        const webgpuInit = await initWebGPU();
        if (!webgpuInit) {
          setError("WebGPU not supported in this browser");
          return;
        }

        const { device, format } = webgpuInit;

        // Configure canvas context
        const context = configureContext(canvas, device, format);
        if (!context) {
          setError("Failed to configure WebGPU context");
          return;
        }

        // Create renderer
        const renderer = new WebGPURenderer({
          device,
          context,
          format,
          canvas,
        });

        rendererRef.current = renderer;

        // Load torus knot geometry
        const torusKnot = generateTorusKnot({
          radius: 1,
          tube: 0.3,
          tubularSegments: 100,
          radialSegments: 16,
          p: 2,
          q: 3,
        });
        renderer.loadGeometry(torusKnot.vertices, torusKnot.indices);

        setIsInitialized(true);

        console.log("WebGPU renderer initialized successfully");
      } catch (err) {
        console.error("Error initializing WebGPU:", err);
        setError(`Initialization error: ${err instanceof Error ? err.message : String(err)}`);
      }
    };

    initializeWebGPU();

    return () => {
      if (rendererRef.current) {
        rendererRef.current.destroy();
      }
    };
  }, []);

  // Render loop with camera updates
  useEffect(() => {
    if (!isInitialized || !rendererRef.current) return;

    let isRunning = true;

    const renderLoop = () => {
      if (!isRunning || !rendererRef.current) return;

      // Calculate camera position from spherical coordinates
      const camera = cameraRef.current;
      const x = camera.distance * Math.sin(camera.phi) * Math.cos(camera.theta);
      const y = camera.distance * Math.cos(camera.phi);
      const z = camera.distance * Math.sin(camera.phi) * Math.sin(camera.theta);

      // Update camera
      rendererRef.current.updateCamera([x, y, z], [0, 0, 0]);

      // Render
      rendererRef.current.render();

      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };

    renderLoop();

    return () => {
      isRunning = false;
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isInitialized]);

  // Handle resize
  useEffect(() => {
    if (!isInitialized || !rendererRef.current || !canvasRef.current) return;

    const handleResize = () => {
      const canvas = canvasRef.current;
      const renderer = rendererRef.current;
      if (!canvas || !renderer) return;

      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = width * devicePixelRatio;
      canvas.height = height * devicePixelRatio;

      renderer.resize(canvas.width, canvas.height);
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [isInitialized]);

  // Mouse controls
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e: MouseEvent) => {
      mouseRef.current.isDragging = true;
      mouseRef.current.lastX = e.clientX;
      mouseRef.current.lastY = e.clientY;
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!mouseRef.current.isDragging) return;

      const deltaX = e.clientX - mouseRef.current.lastX;
      const deltaY = e.clientY - mouseRef.current.lastY;

      // Update camera angles
      cameraRef.current.theta += deltaX * 0.01;
      cameraRef.current.phi -= deltaY * 0.01;

      // Clamp phi to avoid gimbal lock
      cameraRef.current.phi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraRef.current.phi));

      mouseRef.current.lastX = e.clientX;
      mouseRef.current.lastY = e.clientY;
    };

    const handleMouseUp = () => {
      mouseRef.current.isDragging = false;
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      cameraRef.current.distance += e.deltaY * 0.01;
      cameraRef.current.distance = Math.max(2, Math.min(20, cameraRef.current.distance));
    };

    canvas.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    canvas.addEventListener("wheel", handleWheel);

    return () => {
      canvas.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      canvas.removeEventListener("wheel", handleWheel);
    };
  }, []);

  // Keyboard controls (arrow keys)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const step = 0.1;
      switch (e.key) {
        case "ArrowLeft":
          cameraRef.current.theta -= step;
          break;
        case "ArrowRight":
          cameraRef.current.theta += step;
          break;
        case "ArrowUp":
          cameraRef.current.phi = Math.max(0.1, cameraRef.current.phi - step);
          break;
        case "ArrowDown":
          cameraRef.current.phi = Math.min(Math.PI - 0.1, cameraRef.current.phi + step);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  // Shader mode switching
  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setShaderMode(shaderMode);
  }, [shaderMode]);

  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center p-4">
          <p className="text-red-400 mb-2">WebGPU Error</p>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full min-h-[400px] bg-gray-900 rounded-lg overflow-hidden relative">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: "block" }}
      />

      <div className="absolute bottom-4 right-4 text-white text-xs opacity-50">
        WebGPU Renderer {isInitialized ? "(Active)" : "(Initializing...)"}
      </div>

      <div className="absolute top-4 left-4 text-white text-xs opacity-70 bg-black/50 p-2 rounded pointer-events-none">
        Use Arrow Keys to Rotate • Drag to Orbit • Scroll to Zoom
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 bg-black/60 p-3 rounded-lg backdrop-blur-sm">
        {/* WebGL/WebGPU Toggle */}
        {onSwitchToWebGL && (
          <div className="flex flex-col gap-1 pb-2 border-b border-gray-700">
            <label className="text-xs text-gray-300 font-bold">Renderer</label>
            <button
              onClick={onSwitchToWebGL}
              className="bg-green-600 hover:bg-green-500 text-white text-xs py-1 px-2 rounded transition-colors"
            >
              Switch to WebGL
            </button>
          </div>
        )}

        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-300 font-bold">Shader Mode</label>
          <select
            value={shaderMode}
            onChange={(e) => setShaderMode(e.target.value as ShaderMode)}
            className="bg-gray-800 text-white text-xs p-1 rounded border border-gray-700 focus:outline-none focus:border-blue-500"
          >
            <option value="Default">Default (PBR)</option>
            <option value="Normal">Normal</option>
          </select>
        </div>
      </div>

      {!isInitialized && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-white text-sm">Initializing WebGPU...</div>
        </div>
      )}
    </div>
  );
}

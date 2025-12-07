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
  onReset?: () => void;
}

export default function Viewer3DWebGPU({ imageUrl, meshUrl, onSwitchToWebGL, onReset }: Viewer3DWebGPUProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WebGPURenderer | null>(null);
  const animationFrameRef = useRef<number>(0);

  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shaderMode, setShaderMode] = useState<ShaderMode>("Default");

  // ... (lines 25-231)

  // Shader mode switching
  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setShaderMode(shaderMode);
  }, [shaderMode, isInitialized]);

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
        {/* Initialize Button */}
        {(meshUrl || imageUrl) && onReset && (
          <div className="flex flex-col gap-1 pb-2 border-b border-gray-700">
            <label className="text-xs text-gray-300 font-bold">Reset</label>
            <button
              onClick={onReset}
              className="bg-orange-600 hover:bg-orange-500 text-white text-xs py-1 px-2 rounded transition-colors"
            >
              Initialize (Torus Knot)
            </button>
          </div>
        )}

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
            <option value="Toon">Toon</option>
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

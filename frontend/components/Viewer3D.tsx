"use client";

import { Canvas, useLoader } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { Suspense, useEffect, useState, useMemo } from "react";
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";

import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { isWebGPUSupported } from "../lib/webgpu/core/device";
import Viewer3DWebGPU from "./Viewer3DWebGPU";

interface Viewer3DProps {
  imageUrl?: string;
  meshUrl?: string;
}

interface TextureMeshProps {
  url: string;
}

function TextureMesh({ url }: TextureMeshProps) {
  const texture = useLoader(THREE.TextureLoader, url);
  return (
    <mesh>
      <planeGeometry args={[3, 3]} />
      <meshStandardMaterial map={texture} transparent side={THREE.DoubleSide} />
    </mesh>
  );
}

type ShaderMode = "Default" | "Toon" | "Shiny" | "Wireframe" | "Normal";
type LightingPreset = "city" | "sunset" | "studio" | "night";

interface ModelMeshProps {
  url: string;
  shaderMode: ShaderMode;
}

function ModelMesh({ url, shaderMode }: ModelMeshProps) {
  const gltf = useLoader(GLTFLoader, url);

  const scene = useMemo(() => {
    const clonedScene = gltf.scene.clone(true);

    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;

        // Save original material if not already saved
        if (!mesh.userData.originalMaterial) {
          mesh.userData.originalMaterial = mesh.material;
        }

        // Ensure normals exist for shaders that need them
        if (!mesh.geometry.attributes.normal) {
          mesh.geometry.computeVertexNormals();
        }

        const originalMat = mesh.userData.originalMaterial;
        const color = (originalMat as any).color || 0xffffff;
        const map = (originalMat as any).map || null;

        switch (shaderMode) {
          case "Toon":
            mesh.material = new THREE.MeshToonMaterial({
              color: color,
              map: map,
              gradientMap: null, // Could add a gradient map for better toon effect
            });
            break;
          case "Shiny":
            mesh.material = new THREE.MeshStandardMaterial({
              color: color,
              map: map,
              roughness: 0.1,
              metalness: 0.8,
            });
            break;
          case "Wireframe":
            mesh.material = new THREE.MeshBasicMaterial({
              color: 0x00ff00,
              wireframe: true,
            });
            break;
          case "Normal":
            mesh.material = new THREE.MeshNormalMaterial();
            break;
          case "Default":
          default:
            mesh.material = mesh.userData.originalMaterial;
            break;
        }
      }
    });

    return clonedScene;
  }, [gltf.scene, shaderMode]);

  return <primitive object={scene} scale={[2, 2, 2]} />;
}

interface RotatableGroupProps {
  children: React.ReactNode;
}

function RotatableGroup({ children }: RotatableGroupProps) {
  const [rotation, setRotation] = useState<[number, number, number]>([0, 0, 0]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const step = 0.1;
      setRotation((prev) => {
        const [x, y, z] = prev;
        switch (event.key) {
          case "ArrowLeft":
            return [x, y - step, z];
          case "ArrowRight":
            return [x, y + step, z];
          case "ArrowUp":
            return [x - step, y, z];
          case "ArrowDown":
            return [x + step, y, z];
          default:
            return prev;
        }
      });
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return <group rotation={new THREE.Euler(...rotation)}>{children}</group>;
}

export default function Viewer3D({ imageUrl, meshUrl }: Viewer3DProps) {
  const [frameloop, setFrameloop] = useState<"always" | "demand">("always");
  const [shaderMode, setShaderMode] = useState<ShaderMode>("Default");
  const [lightingPreset, setLightingPreset] = useState<LightingPreset>("city");
  const [useWebGPU, setUseWebGPU] = useState(false);
  const [webGPUAvailable, setWebGPUAvailable] = useState(false);

  // Check WebGPU availability
  useEffect(() => {
    setWebGPUAvailable(isWebGPUSupported());
  }, []);

  // Render WebGPU version if enabled
  if (useWebGPU && webGPUAvailable) {
    return <Viewer3DWebGPU imageUrl={imageUrl} meshUrl={meshUrl} onSwitchToWebGL={() => setUseWebGPU(false)} />;
  }

  return (
    <div className="w-full h-full min-h-[400px] bg-gray-900 rounded-lg overflow-hidden relative">
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <Environment preset={lightingPreset} />
        <OrbitControls makeDefault />

        <Suspense fallback={null}>
          <RotatableGroup>
            {meshUrl ? (
              <ModelMesh url={meshUrl} shaderMode={shaderMode} />
            ) : imageUrl ? (
              <TextureMesh url={imageUrl} />
            ) : (
              <mesh>
                <torusKnotGeometry args={[1, 0.3, 100, 16]} />
                <meshNormalMaterial />
              </mesh>
            )}
          </RotatableGroup>
        </Suspense>
      </Canvas>

      <div className="absolute bottom-4 right-4 text-white text-xs opacity-50">
        WebGL Fallback (WebGPU Disabled)
      </div>
      <div className="absolute top-4 left-4 text-white text-xs opacity-70 bg-black/50 p-2 rounded pointer-events-none">
        Use Arrow Keys to Rotate
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 bg-black/60 p-3 rounded-lg backdrop-blur-sm">
        {/* WebGPU Toggle */}
        {webGPUAvailable && (
          <div className="flex flex-col gap-1 pb-2 border-b border-gray-700">
            <label className="text-xs text-gray-300 font-bold">Renderer</label>
            <button
              onClick={() => setUseWebGPU(!useWebGPU)}
              className="bg-blue-600 hover:bg-blue-500 text-white text-xs py-1 px-2 rounded transition-colors"
            >
              Switch to WebGPU
            </button>
          </div>
        )}

        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-300 font-bold">Shader</label>
          <select
            value={shaderMode}
            onChange={(e) => setShaderMode(e.target.value as ShaderMode)}
            className="bg-gray-800 text-white text-xs p-1 rounded border border-gray-700 focus:outline-none focus:border-blue-500"
            suppressHydrationWarning
          >
            <option value="Default">Default</option>
            <option value="Toon">Toon</option>
            <option value="Shiny">Shiny</option>
            <option value="Wireframe">Wireframe</option>
            <option value="Normal">Normal</option>
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-300 font-bold">Lighting</label>
          <select
            value={lightingPreset}
            onChange={(e) => setLightingPreset(e.target.value as LightingPreset)}
            className="bg-gray-800 text-white text-xs p-1 rounded border border-gray-700 focus:outline-none focus:border-blue-500"
            suppressHydrationWarning
          >
            <option value="city">City</option>
            <option value="sunset">Sunset</option>
            <option value="studio">Studio</option>
            <option value="night">Night</option>
          </select>
        </div>
      </div>
    </div>
  );
}

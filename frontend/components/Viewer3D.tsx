import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls, Environment, Center } from "@react-three/drei";
import { Suspense, useEffect, useState, useMemo, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";

import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { isWebGPUSupported } from "../lib/webgpu/core/device";
import Viewer3DWebGPU from "./Viewer3DWebGPU";

export interface Viewer3DRef {
  captureScreenshot: () => string | null;
}

interface Viewer3DProps {
  imageUrl?: string;
  meshUrl?: string;
  onReset?: () => void;
  ambientIntensity?: number;
  directIntensity?: number;
}

interface TextureMeshProps {
  url: string;
}

function TextureMesh({ url }: TextureMeshProps) {
  const texture = useLoader(THREE.TextureLoader, url);
  return (
    <Center>
      <mesh>
        <planeGeometry args={[3, 3]} />
        <meshStandardMaterial map={texture} transparent side={THREE.DoubleSide} />
      </mesh>
    </Center>
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
  const controls = useThree((state) => state.controls as any);
  const camera = useThree((state) => state.camera);

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

  useEffect(() => {
    if (!scene || !camera || !controls) return;

    // Re-target orbit controls to the actual mesh bounds so rotations feel correct
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) {
      return;
    }

    controls.target.copy(center);

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = THREE.MathUtils.degToRad((camera as any).fov || 50);
    const distance = maxDim > 0 ? (maxDim * 1.2) / Math.tan(fov / 2) : 5;
    const offsetDir = new THREE.Vector3(1, 1, 1).normalize();

    camera.position.copy(center.clone().add(offsetDir.multiplyScalar(distance)));
    camera.near = Math.max(0.01, distance / 100);
    camera.far = Math.max(camera.far, distance * 4);
    camera.updateProjectionMatrix();
    controls.update();
  }, [scene, camera, controls, url]);

  return (
    <Center>
      <primitive object={scene} scale={[2, 2, 2]} />
    </Center>
  );
}

interface ScreenShotHandlerProps {
  exposedRef: React.Ref<Viewer3DRef>;
}

// Helper component to access GL context 
const ScreenshotHandler = ({ exposedRef }: ScreenShotHandlerProps) => {
  const { gl, scene, camera } = useThree();

  useImperativeHandle(exposedRef, () => ({
    captureScreenshot: () => {
      gl.render(scene, camera);
      return gl.domElement.toDataURL("image/png");
    }
  }));

  return null;
};

// ... RotatableGroup and TorusKnotMesh omitted for brevity (they don't need changes except maybe Center) ...
// Actually better to include them to be safe if I'm replacing the whole file structure or large chunks.
// To avoid massive replacement, I will edit carefully. The RotatableGroup is fine. 
// I'll replace the main component logic.

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

interface TorusKnotMeshProps {
  shaderMode: ShaderMode;
}

function TorusKnotMesh({ shaderMode }: TorusKnotMeshProps) {
  const material = useMemo(() => {
    switch (shaderMode) {
      case "Toon":
        return <meshToonMaterial color={0x9333ea} />;
      case "Shiny":
        return <meshStandardMaterial color={0x9333ea} roughness={0.1} metalness={0.8} />;
      case "Wireframe":
        return <meshBasicMaterial color={0x00ff00} wireframe />;
      case "Normal":
        return <meshNormalMaterial />;
      case "Default":
      default:
        return <meshStandardMaterial color={0x9333ea} roughness={0.5} metalness={0.3} />;
    }
  }, [shaderMode]);

  return (
    <Center>
      <mesh>
        <torusKnotGeometry args={[1, 0.3, 100, 16]} />
        {material}
      </mesh>
    </Center>
  );
}


const Viewer3D = forwardRef<Viewer3DRef, Viewer3DProps>(({
  imageUrl,
  meshUrl,
  onReset,
  ambientIntensity = 0.5,
  directIntensity = 1.0
}, ref) => {
  const [shaderMode, setShaderMode] = useState<ShaderMode>("Default");
  const [lightingPreset, setLightingPreset] = useState<LightingPreset>("city");
  const [useWebGPU, setUseWebGPU] = useState(false);
  const [webGPUAvailable, setWebGPUAvailable] = useState(false);

  useEffect(() => {
    setWebGPUAvailable(isWebGPUSupported());
  }, []);

  if (useWebGPU && webGPUAvailable) {
    return <Viewer3DWebGPU imageUrl={imageUrl} meshUrl={meshUrl} onSwitchToWebGL={() => setUseWebGPU(false)} onReset={onReset} />;
  }

  return (
    <div className="w-full h-full min-h-[400px] bg-gray-900 rounded-lg overflow-hidden relative">
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        gl={{ preserveDrawingBuffer: true }} // Creating screenshots requires this
      >
        <ScreenshotHandler exposedRef={ref} />
        <ambientLight intensity={ambientIntensity} />
        <directionalLight position={[10, 10, 5]} intensity={directIntensity} />
        <Environment preset={lightingPreset} />
        <OrbitControls makeDefault />

        <Suspense fallback={null}>
          <RotatableGroup>
            {meshUrl ? (
              <ModelMesh url={meshUrl} shaderMode={shaderMode} />
            ) : imageUrl ? (
              <TextureMesh url={imageUrl} />
            ) : (
              <TorusKnotMesh shaderMode={shaderMode} />
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
      <div className="absolute top-4 right-4 flex flex-col gap-2 bg-black/60 p-3 rounded-lg backdrop-blur-sm z-50">
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
          <label className="text-xs text-gray-300 font-bold">Env Light</label>
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
});

Viewer3D.displayName = "Viewer3D";
export default Viewer3D;

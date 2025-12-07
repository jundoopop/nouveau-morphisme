"use client";

import { useState, useRef } from "react";
import Viewer3D from "./Viewer3D";
import LoadingOverlay from "./LoadingOverlay";

export default function IconGenerator() {
    const [file, setFile] = useState<File | null>(null); // Original upload
    const [processedImage, setProcessedImage] = useState<string | null>(null); // Preview of bg-removed
    const [processedBlob, setProcessedBlob] = useState<Blob | null>(null); // Stored bg-removed image for reuse
    const [meshUrl, setMeshUrl] = useState<string | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);

    // Mesh params restored
    const [meshParams, setMeshParams] = useState({
        depth_percentile: 53,
        marching_cubes_threshold: 16.0,
        foreground_ratio: 0.95,
        morph_kernel_size: 3,
        morph_iterations: 1,
        mesh_resolution: 256,
        debug_mode: false,
        use_sam2: true,
        processing_mode: "auto"
    });

    const [lightingParams, setLightingParams] = useState({
        ambient: 0.5,
        direct: 1.0
    });

    // Type checking for Viewer3DRef if needed, but we can just use any or import it
    const viewerRef = useRef<any>(null); // Ideally import Viewer3DRef from Viewer3D

    const generateMesh = async (source: Blob | File, paramsOverride?: Partial<typeof meshParams>) => {
        setIsGenerating(true);
        try {
            const mergedParams = { ...meshParams, ...paramsOverride };
            const meshFormData = new FormData();
            meshFormData.append("file", source);
            meshFormData.append("params", JSON.stringify(mergedParams));

            const meshResponse = await fetch("http://localhost:8000/generate-mesh", {
                method: "POST",
                body: meshFormData
            });
            if (meshResponse.ok) {
                const meshBlob = await meshResponse.blob();
                if (meshUrl) URL.revokeObjectURL(meshUrl);
                const newMeshUrl = URL.createObjectURL(meshBlob);
                setMeshUrl(newMeshUrl);
            } else {
                console.error("Failed to generate mesh");
            }
        } catch (meshError) {
            console.error("Error generating mesh:", meshError);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);

            const formData = new FormData();
            formData.append("file", selectedFile);

            try {
                const response = await fetch("http://localhost:8000/remove-bg", {
                    method: "POST",
                    body: formData,
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    setProcessedImage(url);
                    setProcessedBlob(blob);

                    // Trigger Mesh Generation using original file (kept intact for depth changes)
                    await generateMesh(selectedFile);

                } else {
                    console.error("Failed to remove background");
                }
            } catch (error) {
                console.error("Error uploading file:", error);
            }
        }
    };

    const handleReset = () => {
        // Clean up object URLs to prevent memory leaks
        if (processedImage) {
            URL.revokeObjectURL(processedImage);
        }
        if (meshUrl) {
            URL.revokeObjectURL(meshUrl);
        }

        // Reset all state to initial values
        setFile(null);
        setProcessedImage(null);
        setProcessedBlob(null);
        setMeshUrl(null);
        setIsGenerating(false);
    };

    const handleExport = () => {
        if (viewerRef.current) {
            const dataUrl = viewerRef.current.captureScreenshot();
            if (dataUrl) {
                const link = document.createElement('a');
                link.href = dataUrl;
                link.download = `3d-icon-${Date.now()}.png`;
                link.click();
            }
        }
    };

    const handleDepthChange = async (value: number) => {
        const newParams = { ...meshParams, depth_percentile: value };
        setMeshParams(newParams);
        if (file) {
            // Re-generate mesh using original upload so depth tracking is recalculated
            await generateMesh(file, newParams);
        }
    };

    const handleProcessingModeChange = async (newMode: string) => {
        const newParams = { ...meshParams, processing_mode: newMode };
        setMeshParams(newParams);
        if (file) {
            // Re-generate mesh with new processing mode
            await generateMesh(file, newParams);
        }
    };

    return (
        <div className="flex flex-col md:flex-row h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-black text-white p-4 gap-4 overflow-hidden relative">
            {/* Ambient Background Blobs */}
            <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-purple-600 rounded-full mix-blend-screen filter blur-[120px] opacity-30 animate-pulse"></div>
            <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-blue-600 rounded-full mix-blend-screen filter blur-[120px] opacity-30 animate-pulse delay-1000"></div>

            <div className="w-full md:w-1/3 flex flex-col gap-4 p-6 glass-panel z-10 overflow-y-auto">
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    3D Icon Composer
                </h1>

                <div className="flex flex-col gap-2">
                    <label className="text-sm font-medium text-gray-400">1. Upload Image</label>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleUpload}
                        className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
                    />
                </div>

                {file && (
                    <div className="p-3 glass-dark rounded-lg flex items-center justify-between border border-white/10">
                        <p className="text-xs text-blue-200">Selected: {file.name}</p>
                    </div>
                )}

                <div className="flex flex-col gap-4">
                    <label className="text-sm font-medium text-gray-400">2. Lighting Settings</label>

                    <div className="flex flex-col gap-1">
                        <div className="flex justify-between text-xs text-gray-300">
                            <span>Ambient Intensity</span>
                            <span>{lightingParams.ambient.toFixed(1)}</span>
                        </div>
                        <input
                            type="range"
                            min="0" max="2" step="0.1"
                            value={lightingParams.ambient}
                            onChange={(e) => setLightingParams({ ...lightingParams, ambient: parseFloat(e.target.value) })}
                            className="w-full accent-blue-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>

                    <div className="flex flex-col gap-1">
                        <div className="flex justify-between text-xs text-gray-300">
                            <span>Direct Intensity</span>
                            <span>{lightingParams.direct.toFixed(1)}</span>
                        </div>
                        <input
                            type="range"
                            min="0" max="5" step="0.1"
                            value={lightingParams.direct}
                            onChange={(e) => setLightingParams({ ...lightingParams, direct: parseFloat(e.target.value) })}
                            className="w-full accent-purple-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>
                </div>

                {/* Background Removal Mode */}
                <div className="flex flex-col gap-2 mt-2">
                    <label className="text-sm font-medium text-gray-400">3. Background Removal</label>
                    <div className="p-3 bg-gray-800/50 rounded-lg space-y-3">
                        <div className="flex flex-col gap-2">
                            <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700/30 p-2 rounded transition-colors">
                                <input
                                    type="radio"
                                    name="processing_mode"
                                    value="auto"
                                    checked={meshParams.processing_mode === "auto"}
                                    onChange={(e) => handleProcessingModeChange(e.target.value)}
                                    className="accent-blue-500"
                                />
                                <div className="flex-1">
                                    <div className="text-xs font-semibold text-blue-300">Auto (Recommended)</div>
                                    <div className="text-xs text-gray-400">Detects transparency quality and chooses best mode</div>
                                </div>
                            </label>

                            <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700/30 p-2 rounded transition-colors">
                                <input
                                    type="radio"
                                    name="processing_mode"
                                    value="conservative"
                                    checked={meshParams.processing_mode === "conservative"}
                                    onChange={(e) => handleProcessingModeChange(e.target.value)}
                                    className="accent-green-500"
                                />
                                <div className="flex-1">
                                    <div className="text-xs font-semibold text-green-300">Conservative</div>
                                    <div className="text-xs text-gray-400">Preserves fine details, fuzzy edges OK</div>
                                </div>
                            </label>

                            <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-700/30 p-2 rounded transition-colors">
                                <input
                                    type="radio"
                                    name="processing_mode"
                                    value="aggressive"
                                    checked={meshParams.processing_mode === "aggressive"}
                                    onChange={(e) => handleProcessingModeChange(e.target.value)}
                                    className="accent-orange-500"
                                />
                                <div className="flex-1">
                                    <div className="text-xs font-semibold text-orange-300">Aggressive</div>
                                    <div className="text-xs text-gray-400">Removes all background, may erode surfaces</div>
                                </div>
                            </label>
                        </div>

                        {meshParams.processing_mode === "aggressive" && (
                            <div className="bg-orange-900/20 border border-orange-500/30 rounded p-2 text-xs text-orange-200">
                                <span className="font-semibold">⚠ Warning:</span> Aggressive mode uses binary alpha. Fine details may be lost.
                            </div>
                        )}
                    </div>
                </div>

                {/* Mesh Advanced Settings (Existing logic - implied but not fully shown in original, adding placeholder wrapper) */}
                <div className="flex flex-col gap-2 mt-2">
                    <label className="text-sm font-medium text-gray-400">4. Advanced Mesh</label>
                    <div className="p-3 bg-gray-800/50 rounded text-xs text-gray-400">
                        {/* Ideally we would map meshParams here, but let's just keep the placeholder for now or minimal controls
                            since user asked about lighting specifically.
                        */}
                        <div className="flex justify-between mb-1">
                            <span>Depth Percentile</span>
                            <span>{meshParams.depth_percentile}</span>
                        </div>
                        <input
                            type="range"
                            min="20" max="90" step="1"
                            value={meshParams.depth_percentile}
                            onChange={(e) => handleDepthChange(parseInt(e.target.value))}
                            className="w-full accent-green-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>
                </div>


                <button
                    onClick={handleExport}
                    className="mt-auto bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded-lg transition-colors shadow-lg shadow-blue-600/20"
                >
                    Export Icon (Screenshot)
                </button>
            </div>

            <div className="w-full md:w-2/3 glass-panel flex items-center justify-center relative z-10 overflow-hidden">
                {isGenerating && <LoadingOverlay />}
                <Viewer3D
                    ref={viewerRef}
                    imageUrl={processedImage || undefined}
                    meshUrl={meshUrl || undefined}
                    onReset={handleReset}
                    ambientIntensity={lightingParams.ambient}
                    directIntensity={lightingParams.direct}
                />
            </div>
        </div >
    );
}

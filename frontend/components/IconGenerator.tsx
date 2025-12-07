"use client";

import { useState } from "react";
import Viewer3D from "./Viewer3D";

export default function IconGenerator() {
    const [file, setFile] = useState<File | null>(null);
    const [processedImage, setProcessedImage] = useState<string | null>(null);
    const [meshUrl, setMeshUrl] = useState<string | null>(null);

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

                    // Trigger Mesh Generation
                    const meshFormData = new FormData();
                    meshFormData.append("file", blob, "processed.png");

                    try {
                        const meshResponse = await fetch("http://localhost:8000/generate-mesh", {
                            method: "POST",
                            body: meshFormData
                        });
                        if (meshResponse.ok) {
                            const meshBlob = await meshResponse.blob();
                            const meshUrl = URL.createObjectURL(meshBlob);
                            setMeshUrl(meshUrl);
                        } else {
                            console.error("Failed to generate mesh");
                        }
                    } catch (meshError) {
                        console.error("Error generating mesh:", meshError);
                    }

                } else {
                    console.error("Failed to remove background");
                }
            } catch (error) {
                console.error("Error uploading file:", error);
            }
        }
    };

    return (
        <div className="flex flex-col md:flex-row h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-black text-white p-4 gap-4 overflow-hidden relative">
            {/* Ambient Background Blobs */}
            <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-purple-600 rounded-full mix-blend-screen filter blur-[120px] opacity-30 animate-pulse"></div>
            <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-blue-600 rounded-full mix-blend-screen filter blur-[120px] opacity-30 animate-pulse delay-1000"></div>

            <div className="w-full md:w-1/3 flex flex-col gap-4 p-6 glass-panel z-10">
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

                <div className="flex flex-col gap-2">
                    <label className="text-sm font-medium text-gray-400">2. Settings</label>
                    {/* Controls will go here */}
                    <div className="p-4 bg-gray-900 rounded text-center text-gray-500 text-sm">
                        Controls (Leva)
                    </div>
                </div>

                <button className="mt-auto bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-4 rounded-lg transition-colors">
                    Export Icon Pack
                </button>
            </div>

            <div className="w-full md:w-2/3 glass-panel flex items-center justify-center relative z-10 overflow-hidden">
                <Viewer3D imageUrl={processedImage || undefined} meshUrl={meshUrl || undefined} />
            </div>
        </div >
    );
}

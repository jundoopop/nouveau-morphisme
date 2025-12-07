import React from 'react';

export default function LoadingOverlay() {
    return (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center glass-dark backdrop-blur-md rounded-xl transition-all duration-300">
            <div className="relative w-16 h-16 mb-4">
                {/* Ping effect for attention */}
                <div className="absolute inset-0 border-4 border-blue-500/30 rounded-full animate-ping"></div>
                {/* Spinning gradient border */}
                <div className="absolute inset-0 border-4 border-t-blue-400 border-r-transparent border-b-purple-500 border-l-transparent rounded-full animate-spin"></div>
            </div>
            <p className="text-lg font-medium text-blue-100 animate-pulse bg-gradient-to-r from-blue-200 to-purple-200 bg-clip-text text-transparent">
                Generating 3D Mesh...
            </p>
            <p className="text-xs text-gray-400 mt-2">This may take 15-30 seconds</p>
        </div>
    );
}

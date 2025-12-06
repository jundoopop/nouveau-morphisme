/**
 * WebGPU Device Initialization
 * Handles adapter and device creation with feature detection
 */

export interface WebGPUInitResult {
  device: GPUDevice;
  adapter: GPUAdapter;
  format: GPUTextureFormat;
}

/**
 * Initialize WebGPU adapter and device
 * Returns null if WebGPU is not supported
 */
export async function initWebGPU(): Promise<WebGPUInitResult | null> {
  // Check browser support
  if (!navigator.gpu) {
    console.warn("WebGPU not supported in this browser");
    return null;
  }

  try {
    // Request adapter with high-performance preference
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (!adapter) {
      console.error("Failed to get WebGPU adapter");
      return null;
    }

    // Adapter acquired successfully
    console.log("WebGPU Adapter acquired successfully");

    // Request device with required features and limits
    const device = await adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {
        maxBindGroups: 4,
        maxUniformBufferBindingSize: 65536,
        maxStorageBufferBindingSize: 134217728,
        maxBufferSize: 268435456,
      },
    });

    // Handle device loss
    device.lost.then((info) => {
      console.error(`WebGPU device lost: ${info.message}`);
      if (info.reason === "destroyed") {
        console.log("Device was intentionally destroyed");
      } else {
        console.error("Device lost unexpectedly - may need to reinitialize");
      }
    });

    // Set up error handling
    device.addEventListener("uncapturederror", (event) => {
      console.error("WebGPU uncaptured error:", event.error);
    });

    // Get preferred canvas format
    const format = navigator.gpu.getPreferredCanvasFormat();

    console.log("WebGPU initialized successfully", {
      format,
      limits: device.limits,
    });

    return { device, adapter, format };
  } catch (error) {
    console.error("Error initializing WebGPU:", error);
    return null;
  }
}

/**
 * Check if WebGPU is supported in the current browser
 */
export function isWebGPUSupported(): boolean {
  return "gpu" in navigator;
}

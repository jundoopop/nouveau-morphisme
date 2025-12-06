/**
 * WebGPU Canvas Context Configuration
 * Handles context creation and configuration for rendering
 */

/**
 * Configure WebGPU context for a canvas element
 * @param canvas - The HTML canvas element
 * @param device - The WebGPU device
 * @param format - The texture format to use
 * @returns Configured GPUCanvasContext or null on failure
 */
export function configureContext(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  format: GPUTextureFormat
): GPUCanvasContext | null {
  const context = canvas.getContext("webgpu");

  if (!context) {
    console.error("Failed to get WebGPU context from canvas");
    return null;
  }

  try {
    context.configure({
      device,
      format,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    console.log("WebGPU context configured:", { format });

    return context;
  } catch (error) {
    console.error("Error configuring WebGPU context:", error);
    return null;
  }
}

/**
 * Create a depth texture for depth testing
 * @param device - The WebGPU device
 * @param width - Texture width
 * @param height - Texture height
 * @returns GPU texture for depth testing
 */
export function createDepthTexture(
  device: GPUDevice,
  width: number,
  height: number
): GPUTexture {
  return device.createTexture({
    size: [width, height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

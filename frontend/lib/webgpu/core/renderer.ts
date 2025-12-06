/**
 * WebGPU Renderer
 * Main rendering engine that manages pipelines, buffers, and the render loop
 */

import { mat4 } from "gl-matrix";
import { vertexShader } from "../shaders/vertex.wgsl";
import { fragmentShaderDefault } from "../shaders/fragment-default.wgsl";
import { fragmentShaderNormal } from "../shaders/fragment-normal.wgsl";

export type ShaderMode = "Default" | "Normal";

export interface RendererOptions {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

export class WebGPURenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  private depthTexture: GPUTexture;
  private pipelines: Map<ShaderMode, GPURenderPipeline> = new Map();
  private currentPipeline: GPURenderPipeline | null = null;
  private currentShaderMode: ShaderMode = "Default";

  private uniformBuffers: {
    model: GPUBuffer;
    camera: GPUBuffer;
    light: GPUBuffer;
    material: GPUBuffer;
  };

  private bindGroups: GPUBindGroup[] = [];

  // Geometry data
  private vertexBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private indexCount: number = 0;

  constructor(options: RendererOptions) {
    this.device = options.device;
    this.context = options.context;
    this.format = options.format;
    this.canvas = options.canvas;

    // Create depth texture
    this.depthTexture = this.createDepthTexture(
      this.canvas.width,
      this.canvas.height
    );

    // Initialize uniform buffers
    this.uniformBuffers = this.createUniformBuffers();

    // Initialize with test triangle (will be replaced by loadGeometry)
    this.createTestTriangle();

    // Create all pipelines
    this.createRenderPipelines();
  }

  private createDepthTexture(width: number, height: number): GPUTexture {
    return this.device.createTexture({
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private createUniformBuffers() {
    return {
      model: this.device.createBuffer({
        size: 128, // 2 * mat4x4<f32>
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      camera: this.device.createBuffer({
        size: 144, // 2 * mat4x4<f32> + vec3<f32> + padding
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      light: this.device.createBuffer({
        size: 48, // 3 * vec4<f32>
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      material: this.device.createBuffer({
        size: 32, // vec4<f32> + 2 * f32 + padding
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  private createTestTriangle() {
    // Simple triangle vertices (position, normal, uv, barycentric)
    const vertices = new Float32Array([
      // Position        Normal          UV       Barycentric
      0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0, // Top
      -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // Bottom left
      0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Bottom right
    ]);

    const indices = new Uint32Array([0, 1, 2]);

    // Create vertex buffer
    this.vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    // Create index buffer
    this.indexBuffer = this.device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.indexCount = indices.length;
  }

  private createRenderPipelines() {
    // Create shader modules
    const vertexModule = this.device.createShaderModule({
      code: vertexShader,
      label: "Vertex Shader",
    });

    const fragmentModules = {
      Default: this.device.createShaderModule({
        code: fragmentShaderDefault,
        label: "Fragment Shader - Default",
      }),
      Normal: this.device.createShaderModule({
        code: fragmentShaderNormal,
        label: "Fragment Shader - Normal",
      }),
    };

    // Create bind group layouts
    const bindGroupLayout0 = this.device.createBindGroupLayout({
      label: "Bind Group Layout 0 - Model & Camera",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const bindGroupLayout1 = this.device.createBindGroupLayout({
      label: "Bind Group Layout 1 - Light & Material",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const bindGroupLayout2 = this.device.createBindGroupLayout({
      label: "Bind Group Layout 2 - Textures",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        },
      ],
    });

    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      label: "Pipeline Layout",
      bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1, bindGroupLayout2],
    });

    // Shared vertex state
    const vertexState = {
      module: vertexModule,
      entryPoint: "main",
      buffers: [
        {
          arrayStride: 44, // 11 floats * 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat }, // position
            { shaderLocation: 1, offset: 12, format: "float32x3" as GPUVertexFormat }, // normal
            { shaderLocation: 2, offset: 24, format: "float32x2" as GPUVertexFormat }, // uv
            { shaderLocation: 3, offset: 32, format: "float32x3" as GPUVertexFormat }, // barycentric
          ],
        },
      ],
    };

    // Shared primitive state
    const primitiveState = {
      topology: "triangle-list" as GPUPrimitiveTopology,
      cullMode: "back" as GPUCullMode,
    };

    // Shared depth stencil state
    const depthStencilState = {
      format: "depth24plus" as GPUTextureFormat,
      depthWriteEnabled: true,
      depthCompare: "less" as GPUCompareFunction,
    };

    // Create pipelines for each shader mode
    for (const [mode, fragmentModule] of Object.entries(fragmentModules)) {
      const pipeline = this.device.createRenderPipeline({
        label: `Render Pipeline - ${mode}`,
        layout: pipelineLayout,
        vertex: vertexState,
        fragment: {
          module: fragmentModule,
          entryPoint: "main",
          targets: [
            {
              format: this.format,
              blend: {
                color: {
                  srcFactor: "src-alpha",
                  dstFactor: "one-minus-src-alpha",
                  operation: "add",
                },
                alpha: {
                  srcFactor: "one",
                  dstFactor: "one-minus-src-alpha",
                  operation: "add",
                },
              },
            },
          ],
        },
        primitive: primitiveState,
        depthStencil: depthStencilState,
      });

      this.pipelines.set(mode as ShaderMode, pipeline);
    }

    // Set initial pipeline
    this.currentPipeline = this.pipelines.get("Default")!;

    // Create bind groups
    this.createBindGroups(bindGroupLayout0, bindGroupLayout1, bindGroupLayout2);

    // Initialize uniforms with default values
    this.initializeUniforms();
  }

  private createBindGroups(
    layout0: GPUBindGroupLayout,
    layout1: GPUBindGroupLayout,
    layout2: GPUBindGroupLayout
  ) {
    // Bind group 0: Model & Camera
    this.bindGroups[0] = this.device.createBindGroup({
      label: "Bind Group 0",
      layout: layout0,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffers.model } },
        { binding: 1, resource: { buffer: this.uniformBuffers.camera } },
      ],
    });

    // Bind group 1: Light & Material
    this.bindGroups[1] = this.device.createBindGroup({
      label: "Bind Group 1",
      layout: layout1,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffers.light } },
        { binding: 1, resource: { buffer: this.uniformBuffers.material } },
      ],
    });

    // Bind group 2: Textures (create default 1x1 white texture)
    const defaultTexture = this.createDefaultTexture();
    const defaultSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    this.bindGroups[2] = this.device.createBindGroup({
      label: "Bind Group 2",
      layout: layout2,
      entries: [
        { binding: 0, resource: defaultTexture.createView() },
        { binding: 1, resource: defaultSampler },
      ],
    });
  }

  private createDefaultTexture(): GPUTexture {
    const texture = this.device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );

    return texture;
  }

  private initializeUniforms() {
    // Model matrix (identity)
    const modelMatrix = mat4.create();
    const normalMatrix = mat4.create();
    const modelData = new Float32Array(32);
    modelData.set(modelMatrix, 0);
    modelData.set(normalMatrix, 16);
    this.device.queue.writeBuffer(this.uniformBuffers.model, 0, modelData);

    // Camera matrices
    const viewMatrix = mat4.create();
    const projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 100);
    mat4.lookAt(viewMatrix, [0, 0, 3], [0, 0, 0], [0, 1, 0]);

    const cameraData = new Float32Array(36);
    cameraData.set(viewMatrix, 0);
    cameraData.set(projectionMatrix, 16);
    cameraData.set([0, 0, 3], 32); // camera position
    this.device.queue.writeBuffer(this.uniformBuffers.camera, 0, cameraData);

    // Light data
    const lightData = new Float32Array(12);
    lightData.set([0.2, 0.2, 0.2], 0); // ambient color
    lightData[3] = 1.0; // ambient intensity
    lightData.set([0, -1, 0], 4); // directional direction
    lightData.set([1.0, 1.0, 1.0], 8); // directional color
    lightData[11] = 1.0; // directional intensity
    this.device.queue.writeBuffer(this.uniformBuffers.light, 0, lightData);

    // Material data
    const materialData = new Float32Array(8);
    materialData.set([1.0, 0.5, 0.2, 1.0], 0); // base color (orange)
    materialData[4] = 0.5; // roughness
    materialData[5] = 0.1; // metalness
    this.device.queue.writeBuffer(this.uniformBuffers.material, 0, materialData);
  }

  /**
   * Switch shader mode
   */
  setShaderMode(mode: ShaderMode) {
    const pipeline = this.pipelines.get(mode);
    if (pipeline) {
      this.currentPipeline = pipeline;
      this.currentShaderMode = mode;
      console.log(`Switched to shader mode: ${mode}`);
    } else {
      console.warn(`Shader mode not found: ${mode}`);
    }
  }

  /**
   * Update camera position and view matrix
   */
  updateCamera(cameraPosition: [number, number, number], lookAt: [number, number, number] = [0, 0, 0]) {
    const viewMatrix = mat4.create();
    const projectionMatrix = mat4.create();

    mat4.perspective(projectionMatrix, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 100);
    mat4.lookAt(viewMatrix, cameraPosition, lookAt, [0, 1, 0]);

    const cameraData = new Float32Array(36);
    cameraData.set(viewMatrix, 0);
    cameraData.set(projectionMatrix, 16);
    cameraData.set(cameraPosition, 32);
    this.device.queue.writeBuffer(this.uniformBuffers.camera, 0, cameraData);
  }

  /**
   * Load new geometry into the renderer
   * Replaces existing vertex and index buffers
   */
  loadGeometry(vertices: Float32Array, indices: Uint32Array) {
    // Destroy old buffers if they exist
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
    }
    if (this.indexBuffer) {
      this.indexBuffer.destroy();
    }

    // Create new vertex buffer
    this.vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    // Create new index buffer
    this.indexBuffer = this.device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.indexCount = indices.length;

    console.log(`Loaded geometry: ${vertices.length / 11} vertices, ${indices.length / 3} triangles`);
  }

  /**
   * Render a single frame
   */
  render() {
    if (!this.currentPipeline || !this.vertexBuffer || !this.indexBuffer) return;

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(this.currentPipeline);

    // Set bind groups
    this.bindGroups.forEach((bindGroup, index) => {
      renderPass.setBindGroup(index, bindGroup);
    });

    // Set vertex and index buffers
    renderPass.setVertexBuffer(0, this.vertexBuffer);
    renderPass.setIndexBuffer(this.indexBuffer, "uint32");

    // Draw
    renderPass.drawIndexed(this.indexCount);

    renderPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Handle canvas resize
   */
  resize(width: number, height: number) {
    this.depthTexture.destroy();
    this.depthTexture = this.createDepthTexture(width, height);

    // Update projection matrix
    const projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, Math.PI / 4, width / height, 0.1, 100);

    const cameraData = new Float32Array(36);
    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, [0, 0, 3], [0, 0, 0], [0, 1, 0]);
    cameraData.set(viewMatrix, 0);
    cameraData.set(projectionMatrix, 16);
    cameraData.set([0, 0, 3], 32);
    this.device.queue.writeBuffer(this.uniformBuffers.camera, 0, cameraData);
  }

  /**
   * Cleanup resources
   */
  destroy() {
    this.depthTexture.destroy();
    Object.values(this.uniformBuffers).forEach((buffer) => buffer.destroy());
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
  }
}

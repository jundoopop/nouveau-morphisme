/**
 * Shared Vertex Shader (WGSL)
 * Used by all shader modes - transforms vertices and passes data to fragment shaders
 */

export const vertexShader = `
struct ModelUniforms {
  modelMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
}

struct CameraUniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  cameraPosition: vec3<f32>,
}

@group(0) @binding(0) var<uniform> model: ModelUniforms;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) barycentric: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPosition: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) viewDirection: vec3<f32>,
  @location(4) barycentric: vec3<f32>,
}

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;

  // Transform position to world space
  let worldPos = model.modelMatrix * vec4<f32>(input.position, 1.0);
  output.worldPosition = worldPos.xyz;

  // Transform normal to world space
  output.worldNormal = normalize((model.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);

  // Transform to clip space
  output.position = camera.projectionMatrix * camera.viewMatrix * worldPos;

  // Pass UV coordinates
  output.uv = input.uv;

  // Calculate view direction
  output.viewDirection = normalize(camera.cameraPosition - worldPos.xyz);

  // Pass barycentric coordinates for wireframe rendering
  output.barycentric = input.barycentric;

  return output;
}
`;

/**
 * Normal Visualization Fragment Shader (WGSL)
 * Displays surface normals as RGB colors (like Three.js MeshNormalMaterial)
 */

export const fragmentShaderNormal = `
struct FragmentInput {
  @location(0) worldPosition: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) viewDirection: vec3<f32>,
  @location(4) barycentric: vec3<f32>,
}

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
  let normal = normalize(input.worldNormal);

  // Convert normal from range [-1, 1] to color range [0, 1]
  let color = normal * 0.5 + 0.5;

  return vec4<f32>(color, 1.0);
}
`;

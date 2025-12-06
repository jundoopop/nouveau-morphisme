/**
 * Physically-Based Rendering (PBR) Fragment Shader (WGSL)
 * Implements Cook-Torrance microfacet BRDF for realistic material rendering
 * Based on the Disney PBR model with proper energy conservation
 */

import { pbrFunctions } from './common/pbr-functions.wgsl';

export const fragmentShaderDefault = `
${pbrFunctions}

struct LightUniforms {
  ambientColor: vec3<f32>,
  ambientIntensity: f32,
  directionalDirection: vec3<f32>,
  directionalPadding: f32,
  directionalColor: vec3<f32>,
  directionalIntensity: f32,
}

struct MaterialUniforms {
  baseColor: vec4<f32>,
  roughness: f32,
  metalness: f32,
}

@group(1) @binding(0) var<uniform> light: LightUniforms;
@group(1) @binding(1) var<uniform> material: MaterialUniforms;
@group(2) @binding(0) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(1) var baseColorSampler: sampler;

struct FragmentInput {
  @location(0) worldPosition: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) viewDirection: vec3<f32>,
  @location(4) barycentric: vec3<f32>,
}

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
  // Normalize input vectors
  let N = normalize(input.worldNormal);
  let V = normalize(input.viewDirection);
  let L = normalize(-light.directionalDirection);
  let H = normalize(V + L);

  // Sample and convert base color texture from sRGB to linear
  var albedo = material.baseColor;
  let texColor = textureSample(baseColorTexture, baseColorSampler, input.uv);
  albedo = albedo * vec4<f32>(sRGBToLinear(texColor.rgb), texColor.a);

  // Material properties
  let roughness = clamp(material.roughness, 0.04, 1.0); // Clamp to avoid division by zero
  let metallic = clamp(material.metalness, 0.0, 1.0);

  // Calculate base reflectivity (F0) based on metalness
  // Metals use albedo as F0, non-metals use ~0.04
  let F0 = calculateF0(albedo.rgb, metallic);

  // ========================================
  // Cook-Torrance BRDF Calculation
  // ========================================

  // Calculate dot products
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 0.0);
  let HdotV = max(dot(H, V), 0.0);

  // D: Normal Distribution Function (GGX)
  // Determines how many microfacets are aligned with the half vector
  let D = DistributionGGX(N, H, roughness);

  // G: Geometry Function (Smith's method with Schlick-GGX)
  // Models self-shadowing of microfacets
  let G = GeometrySmith(N, V, L, roughness);

  // F: Fresnel (Schlick approximation)
  // Determines reflection vs refraction based on viewing angle
  let F = fresnelSchlick(HdotV, F0);

  // Cook-Torrance specular BRDF
  let numerator = D * G * F;
  let denominator = 4.0 * NdotV * NdotL + 0.0001; // Add epsilon to prevent divide by zero
  let specular = numerator / denominator;

  // ========================================
  // Energy Conservation
  // ========================================

  // kS: Specular contribution (equal to Fresnel)
  let kS = F;

  // kD: Diffuse contribution (what's left after specular reflection)
  // Metals have no diffuse reflection (metallic kills diffuse)
  let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

  // Lambertian diffuse BRDF
  let diffuse = kD * albedo.rgb / PI;

  // ========================================
  // Lighting Calculation
  // ========================================

  // Directional light radiance
  let radiance = light.directionalColor * light.directionalIntensity;

  // Outgoing reflectance (Lo)
  let Lo = (diffuse + specular) * radiance * NdotL;

  // Ambient lighting (IBL would go here in a full implementation)
  let ambient = light.ambientColor * light.ambientIntensity * albedo.rgb;

  // Final HDR color
  let hdrColor = ambient + Lo;

  // ========================================
  // Tonemapping & Gamma Correction
  // ========================================

  // Apply ACES tonemapping to map HDR to LDR
  let toneMapped = ACESFilm(hdrColor);

  // Convert from linear to sRGB for display
  let srgb = linearToSRGB(toneMapped);

  return vec4<f32>(srgb, albedo.a);
}
`;

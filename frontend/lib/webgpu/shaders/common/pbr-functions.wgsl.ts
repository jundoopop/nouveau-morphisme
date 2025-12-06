/**
 * Physically-Based Rendering (PBR) Math Functions (WGSL)
 * Implementation of Cook-Torrance microfacet BRDF
 * Based on the Disney PBR model and standard PBR practices
 */

export const pbrFunctions = `
const PI = 3.14159265359;

// Fresnel-Schlick approximation
// Calculates the Fresnel reflection at grazing angles
// F0: Base reflectivity at normal incidence (0° angle)
// cosTheta: Dot product between view direction and half vector
fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX/Trowbridge-Reitz Normal Distribution Function (NDF)
// Determines the distribution of microfacet normals
// Models how many microfacets are aligned with the half vector
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;

  let nom = a2;
  var denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;

  return nom / denom;
}

// Schlick-GGX Geometry Function
// Models self-shadowing/masking of microfacets
fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = (roughness + 1.0);
  let k = (r * r) / 8.0;

  let nom = NdotV;
  let denom = NdotV * (1.0 - k) + k;

  return nom / denom;
}

// Smith's Geometry Shadowing Function
// Combines geometry obstruction and shadowing
fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);
  let ggx2 = GeometrySchlickGGX(NdotV, roughness);
  let ggx1 = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

// Calculate F0 (base reflectivity) based on metalness
// Metals have colored reflectivity, non-metals ~0.04 (4%)
fn calculateF0(albedo: vec3<f32>, metallic: f32) -> vec3<f32> {
  let dielectric = vec3<f32>(0.04); // Common F0 for dielectrics
  return mix(dielectric, albedo, metallic);
}

// ACES Filmic Tonemapping
// Maps HDR colors to LDR (0-1 range) while preserving detail
fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

// Linear to sRGB gamma correction
// Converts linear light values to sRGB color space for display
fn linearToSRGB(linear: vec3<f32>) -> vec3<f32> {
  return pow(linear, vec3<f32>(1.0 / 2.2));
}

// sRGB to Linear conversion
// Converts sRGB texture samples to linear space for lighting calculations
fn sRGBToLinear(srgb: vec3<f32>) -> vec3<f32> {
  return pow(srgb, vec3<f32>(2.2));
}
`;

/**
 * Torus Knot Geometry Generator
 * Generates parametric torus knot geometry for WebGPU rendering
 */

export interface TorusKnotGeometry {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
}

export interface TorusKnotParams {
  radius?: number;
  tube?: number;
  tubularSegments?: number;
  radialSegments?: number;
  p?: number; // Torus knot parameter p
  q?: number; // Torus knot parameter q
}

/**
 * Generate a torus knot geometry with interleaved vertex data
 * Vertex format: position(3) + normal(3) + uv(2) + barycentric(3) = 11 floats per vertex
 */
export function generateTorusKnot(params: TorusKnotParams = {}): TorusKnotGeometry {
  const {
    radius = 1,
    tube = 0.3,
    tubularSegments = 100,
    radialSegments = 16,
    p = 2,
    q = 3,
  } = params;

  const vertices: number[] = [];
  const indices: number[] = [];

  // Helper function to calculate position on the torus knot curve
  function calculatePositionOnCurve(u: number): [number, number, number] {
    const cu = Math.cos(u);
    const su = Math.sin(u);
    const quOverP = (q / p) * u;
    const cs = Math.cos(quOverP);

    const tx = radius * (2 + cs) * 0.5 * cu;
    const ty = radius * (2 + cs) * su * 0.5;
    const tz = radius * Math.sin(quOverP) * 0.5;

    return [tx, ty, tz];
  }

  // Helper function to calculate tangent (derivative of position)
  function calculateTangent(u: number): [number, number, number] {
    const delta = 0.0001;
    const [x1, y1, z1] = calculatePositionOnCurve(u - delta);
    const [x2, y2, z2] = calculatePositionOnCurve(u + delta);

    const dx = x2 - x1;
    const dy = y2 - y1;
    const dz = z2 - z1;

    // Normalize
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return [dx / len, dy / len, dz / len];
  }

  // Generate vertices
  for (let i = 0; i <= tubularSegments; i++) {
    const u = (i / tubularSegments) * p * Math.PI * 2;

    const [px, py, pz] = calculatePositionOnCurve(u);
    const [tx, ty, tz] = calculateTangent(u);

    // Calculate normal and binormal using Frenet frame
    const nx = -ty;
    const ny = tx;
    const nz = 0;
    const nlen = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const normalX = nlen > 0.00001 ? nx / nlen : 0;
    const normalY = nlen > 0.00001 ? ny / nlen : 1;
    const normalZ = nlen > 0.00001 ? nz / nlen : 0;

    const binormalX = ty * normalZ - tz * normalY;
    const binormalY = tz * normalX - tx * normalZ;
    const binormalZ = tx * normalY - ty * normalX;

    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * Math.PI * 2;
      const cv = Math.cos(v);
      const sv = Math.sin(v);

      // Position
      const posX = px + tube * (cv * normalX + sv * binormalX);
      const posY = py + tube * (cv * normalY + sv * binormalY);
      const posZ = pz + tube * (cv * normalZ + sv * binormalZ);

      // Normal
      const normX = cv * normalX + sv * binormalX;
      const normY = cv * normalY + sv * binormalY;
      const normZ = cv * normalZ + sv * binormalZ;

      // UV coordinates
      const uvU = i / tubularSegments;
      const uvV = j / radialSegments;

      // Barycentric coordinates (will be set per triangle later)
      const baryU = 0;
      const baryV = 0;
      const baryW = 0;

      // Add vertex data (11 floats)
      vertices.push(
        posX, posY, posZ,       // position
        normX, normY, normZ,    // normal
        uvU, uvV,               // uv
        baryU, baryV, baryW     // barycentric (placeholder)
      );
    }
  }

  // Generate indices and barycentric coordinates
  const verticesPerSegment = radialSegments + 1;
  const vertexArray = new Float32Array(vertices);

  for (let i = 0; i < tubularSegments; i++) {
    for (let j = 0; j < radialSegments; j++) {
      const a = i * verticesPerSegment + j;
      const b = ((i + 1) % (tubularSegments + 1)) * verticesPerSegment + j;
      const c = ((i + 1) % (tubularSegments + 1)) * verticesPerSegment + (j + 1);
      const d = i * verticesPerSegment + (j + 1);

      // First triangle (a, b, d)
      indices.push(a, b, d);

      // Set barycentric coordinates for first triangle
      // Vertex a: (1, 0, 0)
      vertexArray[a * 11 + 8] = 1.0;
      vertexArray[a * 11 + 9] = 0.0;
      vertexArray[a * 11 + 10] = 0.0;

      // Vertex b: (0, 1, 0)
      vertexArray[b * 11 + 8] = 0.0;
      vertexArray[b * 11 + 9] = 1.0;
      vertexArray[b * 11 + 10] = 0.0;

      // Vertex d: (0, 0, 1)
      vertexArray[d * 11 + 8] = 0.0;
      vertexArray[d * 11 + 9] = 0.0;
      vertexArray[d * 11 + 10] = 1.0;

      // Second triangle (b, c, d)
      indices.push(b, c, d);

      // Set barycentric coordinates for second triangle
      // Vertex b: (1, 0, 0)
      vertexArray[b * 11 + 8] = 1.0;
      vertexArray[b * 11 + 9] = 0.0;
      vertexArray[b * 11 + 10] = 0.0;

      // Vertex c: (0, 1, 0)
      vertexArray[c * 11 + 8] = 0.0;
      vertexArray[c * 11 + 9] = 1.0;
      vertexArray[c * 11 + 10] = 0.0;

      // Vertex d: (0, 0, 1)
      vertexArray[d * 11 + 8] = 0.0;
      vertexArray[d * 11 + 9] = 0.0;
      vertexArray[d * 11 + 10] = 1.0;
    }
  }

  return {
    vertices: vertexArray,
    indices: new Uint32Array(indices),
    vertexCount: vertexArray.length / 11,
    indexCount: indices.length,
  };
}

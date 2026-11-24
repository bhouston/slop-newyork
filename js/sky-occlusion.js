import * as THREE from 'three/webgpu';
import { Fn, Loop, texture, textureLoad, int, ivec2, vec2, vec3, float, max, atan, sin, dot } from 'three/tsl';
import { ISLAND_BOUNDS } from './city-config.js';

// Sky occlusion of the city from a top-down height map: how much of the sky a point sees over the skyline around it.
// The horizon is found in a few compass directions (slices) by marching outward through a max-mip pyramid of the
// height map, and the cosine-weighted share of the sky above it is integrated per slice in closed form (horizon
// mapping, the slice integral of GTAO).

const METERS_PER_TEXEL = 2;
const WIDTH = 2048, DEPTH = 8192; // texels: 4 km x 16 km, Manhattan with a margin
const MAX_HEIGHT = 600; // m, above One WTC
const SLICES = 16;
// m from the point, 4 to 256 in steps of sqrt(2). A step answers for the ground up to the next one and across the
// width of its slice, about 0.4 of its distance each way: it is read at the mip whose texels are a quarter of the
// distance, where the bilinear footprint is that size. Coarser texels reach back into the wall the point is on.
const STEPS = Array.from({ length: 13 }, (_, i) => 4 * 2 ** (i / 2));
const TEXELS_PER_STEP = 4;
const WALL_CLEARANCE = 0.5; // of the distance: 1.5 texels of the coarser of the two mips a step blends
const NORMAL_OFFSET = METERS_PER_TEXEL; // m off the surface, so a wall doesn't occlude itself
const HEIGHT_BIAS = 0.5; // m

// The height map: the colliders of the buildings (their outlines and roof heights) drawn from the lowest to the
// tallest, so the tallest wins where they overlap.
// ponytail: a 2D canvas holds 8 bits, so the height is stored as its square root, rounded down (0.6 m steps at 20 m,
// 4.5 m at One WTC, and never above the roof); render the city from above into a float target if that gets too coarse
const encodeHeight = (h) => Math.floor(255 * Math.sqrt(Math.min(Math.max(h, 0), MAX_HEIGHT) / MAX_HEIGHT));

// Draws the triangles of the opaque meshes under a root into the height map from above, keeping the tallest. For
// models whose colliders are coarser than they are (the landmarks): a tower narrower than its collider box would
// stand inside the box and see no sky. Texels are sampled at their centres, so parts thinner than one are missed.
function rasterizeMeshes(root, heights, originX, originZ) {
  root.updateWorldMatrix(true, true);
  const matrix = new THREE.Matrix4(), v = new THREE.Vector3();
  root.traverse((mesh) => {
    if (!mesh.isMesh || !mesh.visible || mesh.material.transparent) return;
    const position = mesh.geometry.attributes.position, index = mesh.geometry.index;
    const points = new Float32Array(position.count * 3); // in texels (x, z) and metres (y)
    for (let instance = 0; instance < (mesh.isInstancedMesh ? mesh.count : 1); instance++) {
      matrix.copy(mesh.matrixWorld);
      if (mesh.isInstancedMesh) matrix.multiply(new THREE.Matrix4().fromArray(mesh.instanceMatrix.array, instance * 16));
      for (let i = 0; i < position.count; i++) {
        v.fromBufferAttribute(position, i).applyMatrix4(matrix);
        points[i * 3] = (v.x - originX) / METERS_PER_TEXEL;
        points[i * 3 + 1] = (v.z - originZ) / METERS_PER_TEXEL;
        points[i * 3 + 2] = v.y;
      }
      for (let t = 0, n = index ? index.count : position.count; t < n; t += 3) {
        const a = (index ? index.getX(t) : t) * 3, b = (index ? index.getX(t + 1) : t + 1) * 3, c = (index ? index.getX(t + 2) : t + 2) * 3;
        const ax = points[a], az = points[a + 1], bx = points[b], bz = points[b + 1], cx = points[c], cz = points[c + 1];
        const area = (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
        if (Math.abs(area) < 1e-6) continue; // a wall, seen edge-on
        const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), x1 = Math.min(WIDTH - 1, Math.floor(Math.max(ax, bx, cx)));
        const z0 = Math.max(0, Math.floor(Math.min(az, bz, cz))), z1 = Math.min(DEPTH - 1, Math.floor(Math.max(az, bz, cz)));
        for (let z = z0; z <= z1; z++) {
          for (let x = x0; x <= x1; x++) {
            const px = x + 0.5, pz = z + 0.5;
            const wb = ((px - ax) * (cz - az) - (pz - az) * (cx - ax)) / area;
            const wc = ((bx - ax) * (pz - az) - (bz - az) * (px - ax)) / area;
            if (wb < 0 || wc < 0 || wb + wc > 1) continue;
            const h = encodeHeight(points[a + 2] * (1 - wb - wc) + points[b + 2] * wb + points[c + 2] * wc);
            if (h > heights[z * WIDTH + x]) heights[z * WIDTH + x] = h;
          }
        }
      }
    }
  });
}

function createHeightMap(colliders, meshRoot) {
  const originX = (ISLAND_BOUNDS.minX + ISLAND_BOUNDS.maxX) / 2 - WIDTH * METERS_PER_TEXEL / 2;
  const originZ = (ISLAND_BOUNDS.minZ + ISLAND_BOUNDS.maxZ) / 2 - DEPTH * METERS_PER_TEXEL / 2;
  const canvas = document.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = DEPTH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, WIDTH, DEPTH);
  ctx.setTransform(1 / METERS_PER_TEXEL, 0, 0, 1 / METERS_PER_TEXEL, -originX / METERS_PER_TEXEL, -originZ / METERS_PER_TEXEL);
  for (const c of colliders.filter((c) => c.maxY > 0).sort((a, b) => a.maxY - b.maxY)) {
    const v = encodeHeight(c.maxY);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.beginPath();
    if (c.polygon) {
      for (const ring of [c.polygon.outer, ...(c.polygon.holes || [])]) {
        ring.forEach((p, i) => i ? ctx.lineTo(p.x, p.z) : ctx.moveTo(p.x, p.z));
        ctx.closePath();
      }
    } else if (c.obb) {
      const o = c.obb;
      for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        ctx.lineTo(o.cx + sx * o.hx * o.cs + sz * o.hz * o.sn, o.cz - sx * o.hx * o.sn + sz * o.hz * o.cs);
      }
    } else {
      ctx.rect(c.minX, c.minZ, c.maxX - c.minX, c.maxZ - c.minZ);
    }
    ctx.fill('evenodd');
  }
  const rgba = ctx.getImageData(0, 0, WIDTH, DEPTH).data;
  let level = new Uint8Array(WIDTH * DEPTH);
  for (let i = 0; i < level.length; i++) level[i] = rgba[i * 4];
  if (meshRoot) rasterizeMeshes(meshRoot, level, originX, originZ);
  // Max-mip pyramid: a texel holds the tallest of the four below it, so a far step can't miss a thin tower
  const mipmaps = [{ data: level, width: WIDTH, height: DEPTH }];
  for (let w = WIDTH, h = DEPTH; w > 1 || h > 1;) {
    const nw = Math.max(1, w >> 1), nh = Math.max(1, h >> 1);
    const next = new Uint8Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        const x0 = Math.min(2 * x, w - 1), x1 = Math.min(2 * x + 1, w - 1);
        const r0 = Math.min(2 * y, h - 1) * w, r1 = Math.min(2 * y + 1, h - 1) * w;
        next[y * nw + x] = Math.max(level[r0 + x0], level[r0 + x1], level[r1 + x0], level[r1 + x1]);
      }
    }
    mipmaps.push({ data: next, width: nw, height: nh });
    level = next; w = nw; h = nh;
  }
  const map = new THREE.DataTexture(mipmaps[0].data, WIDTH, DEPTH, THREE.RedFormat, THREE.UnsignedByteType);
  map.mipmaps = mipmaps;
  map.generateMipmaps = false;
  map.magFilter = THREE.LinearFilter;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.needsUpdate = true;
  return { map, originX, originZ };
}

// colliders: the buildings whose outline and roof height are their shape. meshRoot: the models to draw as they are.
function createSkyOcclusion(colliders, meshRoot) {
  const { map, originX, originZ } = createHeightMap(colliders, meshRoot);
  const size = vec2(WIDTH * METERS_PER_TEXEL, DEPTH * METERS_PER_TEXEL);

  // The height of the tallest roof in the texel of the given mip level at a world position (xz), in metres
  const heightAt = (xz, level = 0) => {
    const v = texture(map, xz.sub(vec2(originX, originZ)).div(size)).level(level).r;
    return v.mul(v).mul(MAX_HEIGHT);
  };

  // The share of the cosine-weighted irradiance of an open sky that reaches a surface: 1 on an open field facing up,
  // 0.5 on a wall facing open ground.
  // bias: metres added to the height bias, for positions that are only known roughly (rebuilt from a depth buffer)
  const visibility = (positionWorld, normalWorld, bias = 0) => visibilityOf(positionWorld, normalWorld, float(bias));
  const visibilityOf = Fn(([positionWorld, normalWorld, bias]) => {
    const p = positionWorld.add(normalWorld.mul(NORMAL_OFFSET)).toVar();
    const n = normalWorld.toVar();
    const floor = p.y.add(HEIGHT_BIAS).add(bias).toVar(); // what a roof has to rise above to hide any sky
    const sum = float(0).toVar();
    for (let s = 0; s < SLICES; s++) {
      const phi = (s + 0.5) * Math.PI * 2 / SLICES;
      const d = vec2(Math.cos(phi), Math.sin(phi));
      // On a wall, the slices along it are the ones that see up the street, and the footprint of their reads
      // (1.5 texels) would reach back into the wall: they are turned outward by that much
      const reach = d.add(n.xz.mul(WALL_CLEARANCE)).toVar();
      const reachLength = reach.length();
      let tanH = float(0); // the horizon starts flat
      for (const dist of STEPS) {
        const lod = Math.log2(Math.max(1, dist / TEXELS_PER_STEP / METERS_PER_TEXEL));
        tanH = max(tanH, heightAt(p.xz.add(reach.mul(dist)), lod).sub(floor).div(reachLength.mul(dist)));
      }
      const tH = atan(tanH).toVar(); // the elevation of the horizon
      const cosH = tH.cos();
      // the sky above the horizon, weighted by the cosine to the normal: its vertical and its horizontal part
      const slice = n.y.mul(cosH).mul(cosH).mul(0.5)
        .add(dot(n.xz, d).mul(float(Math.PI / 4).sub(tH.mul(0.5)).sub(sin(tH.mul(2)).mul(0.25))));
      sum.addAssign(max(slice, 0));
    }
    return sum.mul(2 / SLICES).min(1);
  });

  // Debug: the same integral by brute force, to check the one above against. 32 slices, the horizon found texel by
  // texel (every 2 m out to 64 m, then every 4 m in the next mip out to 320 m), 3072 reads a pixel: for a still.
  const visibilityReference = Fn(([positionWorld, normalWorld]) => {
    const p = positionWorld.add(normalWorld.mul(NORMAL_OFFSET)).toVar();
    const n = normalWorld.toVar();
    const sum = float(0).toVar();
    const texel = (xz, level) => {
      const v = textureLoad(map, ivec2(xz.sub(vec2(originX, originZ)).div(METERS_PER_TEXEL << level).floor()), int(level)).r;
      return v.mul(v).mul(MAX_HEIGHT);
    };
    Loop(32, ({ i: s }) => {
      const phi = float(s).add(0.5).mul(Math.PI * 2 / 32);
      const d = vec2(phi.cos(), phi.sin()).toVar();
      const tanH = float(0).toVar();
      Loop(32, ({ i }) => {
        const dist = float(i).add(1).mul(2);
        tanH.assign(max(tanH, texel(p.xz.add(d.mul(dist)), 0).sub(p.y).sub(HEIGHT_BIAS).div(dist)));
      });
      Loop(64, ({ i }) => {
        const dist = float(i).add(1).mul(4).add(64);
        tanH.assign(max(tanH, texel(p.xz.add(d.mul(dist)), 1).sub(p.y).sub(HEIGHT_BIAS).div(dist)));
      });
      const tH = atan(tanH).toVar();
      const cosH = tH.cos();
      const slice = n.y.mul(cosH).mul(cosH).mul(0.5)
        .add(dot(n.xz, d).mul(float(Math.PI / 4).sub(tH.mul(0.5)).sub(sin(tH.mul(2)).mul(0.25))));
      sum.addAssign(max(slice, 0));
    });
    return sum.mul(2 / 32).min(1);
  });

  // Debug: the height map from above, 1 km across the screen's height, centred on a world position (north is up)
  const topDown = (screenUV, aspect, centerWorld) => {
    const xz = centerWorld.xz.add(screenUV.sub(0.5).mul(vec2(aspect, 1)).mul(1000));
    return vec3(heightAt(xz).div(300));
  };

  return { map, heightAt, visibility, visibilityReference, topDown };
}

export { createSkyOcclusion };

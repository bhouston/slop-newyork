import * as THREE from 'three/webgpu';
import { Fn, output, pass, mrt, builtinIndirectContext, clearcoat, sample, packNormalToOctahedral, unpackOctahedralToNormal, vec2, vec3, vec4, velocity, rtt, screenUV, screenSize, uniform, getViewPosition, cubeTexture } from 'three/tsl';
import { ssr } from 'three/addons/tsl/display/SSRNode.js';
import { depthAwareDownsample } from 'three/addons/tsl/display/DepthAwareDownsampleNode.js';
import { jointBilateralUpsample } from 'three/addons/tsl/display/JointBilateralUpsample.js';
import { temporalReproject } from 'three/addons/tsl/display/TemporalReprojectNode.js';
import { reprojectedBeauty, previousViewZ } from 'three/addons/tsl/display/ReprojectedBeautyNode.js';
import { opaqueFragment, opaqueFragmentSource } from 'three/addons/tsl/utils/OpaqueFragment.js';
import { outermostNormalView, outermostRoughness, outermostMetalness } from 'three/addons/tsl/utils/OutermostLobe.js';
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';
import { denoise } from 'three/addons/tsl/display/DenoiseNode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { fsr1 } from 'three/addons/tsl/display/FSR1Node.js';
import { createPostProcessing } from './post-processing.js';
import { animatedWebP } from './capture.js';
import { ClusteredLighting } from 'three/addons/lighting/ClusteredLighting.js';
import { getLocationLabel, isTypingInField, ROADS, NYC_LANDMARKS, TELEPORT_LOCATIONS } from './city-config.js';
import { buildCityGrid } from './city-grid.js';
import { buildBoroughs } from './boroughs.js';
import { buildLandmarks } from './landmarks.js';
import { buildCityBlocks } from './buildings.js';
import { buildStreetDetails } from './street-details.js';
import { createTrafficSystem, createHighwayTraffic } from './traffic.js';
import { buildRoadPaint } from './road-paint.js';
import { ThirdPersonControllerManager } from './controllers.js';
import { TouchControls } from './touch-controls.js';
import { GamepadControls } from './gamepad-controls.js';
import { CameraSystem, CAMERA_PRESETS } from './cameras.js';
import { LightingSystem } from './lighting.js';
import { createSkyOcclusion } from './sky-occlusion.js';
import { canyonSkyAO, cityUniforms } from './tsl-materials.js';
import { NYCMapSystem } from './nyc-map.js';
import { setupSceneCulling, updateSceneCulling } from './scene-culling.js';
import { CityAudioSystem } from './audio-system.js';

// Application setup, shader warm-up and frame loop.

async function initApp() {
  const container = document.getElementById('canvas-container');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingStatus = document.getElementById('loading-status');
  const loadingFill = document.getElementById('loading-progress');

  // One progress bar for the whole start-up. Each stage fills a share of it sized after its typical duration (the
  // "[init]" timings logged below, on an M1 MacBook Air; `data` is fetching the code and data before this runs, which
  // the stylesheet animates). The fill glides through a stage as a compositor transition, so it keeps moving while the
  // stage's synchronous work blocks the main thread, at the pace the earlier stages ran at on this machine.
  const STAGE_MS = {
    data: 1800, renderer: 50, lighting: 60, cityGrid: 1800, boroughs: 560, landmarks: 200, blocks: 2100,
    streetDetails: 2200, traffic: 860, roadPaint: 170, controllers: 50, culling: 730, ui: 50, postFX: 60,
    compile: 3900, warmup: 1950, gpu: 300
  };
  const stageNames = Object.keys(STAGE_MS);
  const totalStageMs = stageNames.reduce((sum, name) => sum + STAGE_MS[name], 0);
  const loadingStartedAt = performance.now();
  let loadingStage = { from: 0, to: STAGE_MS.data / totalStageMs };

  const setLoadingProgress = (p, seconds = 0.3) => {
    if (!loadingFill) return;
    loadingFill.style.animation = 'none';
    loadingFill.style.transition = `transform ${seconds}s ${seconds > 0.3 ? 'linear' : 'ease'}`;
    loadingFill.style.transform = `scaleX(${Math.min(1, Math.max(0, p))})`;
  };
  // Progress through the current stage, for stages that can measure it
  const setStageProgress = (f) => setLoadingProgress(loadingStage.from + (loadingStage.to - loadingStage.from) * f);

  function beginLoadingStage(message, name, glide = true) {
    if (loadingStatus) loadingStatus.textContent = message;
    const index = stageNames.indexOf(name);
    const before = stageNames.slice(0, index).reduce((sum, n) => sum + STAGE_MS[n], 0);
    loadingStage = { from: before / totalStageMs, to: (before + STAGE_MS[name]) / totalStageMs };
    if (!glide) return setLoadingProgress(loadingStage.from);
    const ranMs = before - STAGE_MS.data;
    const pace = ranMs >= 1000 ? Math.min(4, Math.max(0.5, (performance.now() - loadingStartedAt) / ranMs)) : 1;
    setLoadingProgress(loadingStage.to, (STAGE_MS[name] * pace) / 1000);
  }

  async function showLoadingStage(message, name, glide) {
    beginLoadingStage(message, name, glide);
    if (document.hidden) return;
    // Let the label paint before synchronous work. The timeout also lets hidden tabs keep loading.
    await new Promise((resolve) => {
      const timeout = setTimeout(resolve, 100);
      requestAnimationFrame(() => {
        clearTimeout(timeout);
        setTimeout(resolve, 0);
      });
    });
  }

  await showLoadingStage('Initializing graphics...', 'renderer');

  // 1. Initialize Three.js WebGPURenderer with 32-bit float reversedDepthBuffer.
  // No MSAA: TRAA anti-aliases the image, which lets the beauty pass test against the pre-pass's depth.
  const renderer = new THREE.WebGPURenderer({
    antialias: false,
    powerPreference: 'high-performance',
    reversedDepthBuffer: true
  });
  // Every vertex shader computes its position the same way whatever else it computes (WGSL's @invariant): the beauty
  // pass tests against the pre-pass's depth, and the two shaders of a material differ in everything but the position.
  // Without it the compiler is free to round the two differently, and the beauty loses fragments to its own depth.
  renderer.contextNode.value.invariant = true;
  renderer.setPixelRatio(1.0);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.lighting = new ClusteredLighting(256, 32, 28, 48);
  container.appendChild(renderer.domElement);

  // Start-up phase timings (logged once as "[init] ...") to keep load-time regressions traceable
  let initTimings = [];
  let initMark = performance.now();
  const markInit = (label) => {
    const now = performance.now();
    initTimings.push(`${label} ${Math.round(now - initMark)}ms`);
    initMark = now;
  };

  await renderer.init();
  markInit('renderer');

  // 2. Scene, atmosphere and lighting
  await showLoadingStage('Creating sky and environment lighting...', 'lighting');
  const scene = new THREE.Scene();
  const lightingSystem = new LightingSystem(scene, renderer);
  markInit('lighting');

  // 3. Build Manhattan Geography, Iconic Landmarks, Blocks, Street Details & Traffic
  await showLoadingStage('Building streets, parks and waterfronts...', 'cityGrid');
  const cityGrid = buildCityGrid(scene);
  markInit('cityGrid');
  await showLoadingStage('Building bridges and surrounding boroughs...', 'boroughs');
  const boroughs = buildBoroughs(scene, { sidewalks: cityGrid.sidewalks });
  markInit('boroughs');
  await showLoadingStage('Building landmarks...', 'landmarks');
  const { group: landmarks, colliders: landmarkColliders } = buildLandmarks(scene);
  markInit('landmarks');
  await showLoadingStage('Generating building footprints and facades...', 'blocks');
  const cityBlocks = buildCityBlocks(scene, landmarkColliders);
  markInit('blocks');
  const allColliders = [...landmarkColliders, ...cityBlocks.colliders, ...boroughs.colliders, ...(cityGrid.colliders || [])];
  // The landmarks are drawn into the height map as they are: their colliders are boxes wider than their towers
  const skyOcclusion = createSkyOcclusion(allColliders.filter((c) => !landmarkColliders.includes(c)), landmarks);

  await showLoadingStage('Placing trees, streetlights and street furniture...', 'streetDetails');
  const streetDetails = buildStreetDetails(scene, allColliders);
  // Everything lit in Manhattan receives shadows, so that a material compiles one shader instead of one per
  // receiveShadow value of the meshes it is on (the boroughs, beyond the shadow map, all go without)
  for (const name of ['ManhattanCityGrid', 'ManhattanLandmarks', 'ManhattanStreetDetails']) {
    scene.getObjectByName(name).traverse((o) => {
      if (o.isMesh) o.receiveShadow = true;
    });
  }
  markInit('streetDetails');
  await showLoadingStage('Creating traffic and pedestrian routes...', 'traffic');
  // Buildings turned along non-grid streets have oriented colliders whose world bounds reach past their lots: they stay
  // inside the property line anyway, so the road network (lanes & sidewalks) doesn't need them.
  // Elevated building sections that clear even a bus don't obstruct ground traffic.
  const trafficSystem = createTrafficSystem(scene, cityGrid.sidewalks, [
    ...cityGrid.footprints,
    ...streetDetails.footprints,
    ...allColliders.filter((c) => !c.obb && (c.minY ?? 0) < 3.5)
  ], streetDetails.signalIntersections);
  markInit('traffic');
  const highwayTraffic = createHighwayTraffic(scene, cityGrid.highwayLanes);
  await showLoadingStage('Adding road markings...', 'roadPaint');
  buildRoadPaint(scene, trafficSystem.network, cityGrid.roadPaint);
  markInit('roadPaint');

  // 4. Initialize Third-Person Controllers (Person, Car, Cyclist, Helicopter) & Multi-Camera System
  await showLoadingStage('Preparing player, vehicles and controls...', 'controllers');
  const controllerManager = new ThirdPersonControllerManager(scene, allColliders);
  // People knocked down land on the same ground as the player and bounce off the same buildings & street furniture
  trafficSystem.setWorld({
    ground: (x, z) => controllerManager.getGroundHeight(x, z, 'person'),
    colliders: (x, z, r) => controllerManager.nearColliders(x, z, r),
    contact: (c, x, y, z, r, out) => controllerManager.sphereContact(c, x, y, z, r, out)
  });
  // Reuse teleport placement so saved players return at rest, close to their last positions.
  try {
    const saved = JSON.parse(localStorage.getItem('nyc-player-state'));
    if (saved?.version === 1) {
      for (const mode of Object.keys(controllerManager.entities)) {
        const p = saved.entities?.[mode];
        if (!p || ![p.x, p.y, p.z, p.heading].every(Number.isFinite)) continue;
        controllerManager.setMode(mode);
        controllerManager.teleportActiveTo(p.x, p.z, p.heading, p.y);
      }
      controllerManager.setMode(Object.hasOwn(controllerManager.entities, saved.activeMode) ? saved.activeMode : 'car');
    }
  } catch { /* Missing or unavailable storage does not prevent playing. */ }

  function savePlayerState() {
    try {
      const entities = Object.fromEntries(Object.entries(controllerManager.entities).map(([mode, entity]) => [mode, {
        x: entity.position.x, y: entity.position.y, z: entity.position.z, heading: entity.heading
      }]));
      localStorage.setItem('nyc-player-state', JSON.stringify({ version: 1, activeMode: controllerManager.activeMode, entities }));
    } catch { /* Storage may be disabled or full. */ }
  }

  const cameraSystem = new CameraSystem(renderer.domElement, controllerManager);
  const touchControls = new TouchControls(controllerManager);
  const soundButton = document.getElementById('btn-sound');
  const audioSystem = new CityAudioSystem({ controllerManager, trafficSystem, highwayTraffic, streetDetails }, {
    onStateChange: updateSoundButton
  });
  function updateSoundButton() {
    if (!soundButton) return;
    soundButton.disabled = !audioSystem.supported;
    soundButton.setAttribute('aria-pressed', String(audioSystem.enabled));
    soundButton.setAttribute('aria-label', audioSystem.enabled ? 'Mute sound' : 'Enable sound');
    soundButton.title = audioSystem.supported ? `${audioSystem.enabled ? 'Mute' : 'Enable'} sound [N]` : 'Web Audio is unavailable';
    soundButton.querySelector('use').setAttribute('href', audioSystem.enabled ? '#i-sound' : '#i-sound-off');
  }
  soundButton?.addEventListener('click', () => audioSystem.setEnabled(!audioSystem.enabled));
  updateSoundButton();
  // pagehide also covers navigation away on mobile; preserve the context across back-forward cache restores.
  window.addEventListener('pagehide', (event) => {
    savePlayerState();
    if (!event.persisted) audioSystem.dispose();
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) savePlayerState(); });
  markInit('controllers');

  // Island-wide static geometry: chunk & distance-cull the small props, shadow stand-ins for the big casters
  // (scene-culling.js)
  await showLoadingStage('Preparing scene visibility and shadows...', 'culling');
  const cullingRoots = ['ManhattanCityGrid', 'ManhattanCityBlocks', 'ManhattanStreetDetails', 'ManhattanLandmarks', 'Boroughs', 'RoadMarkings']
    .map((name) => scene.getObjectByName(name))
    .filter(Boolean);
  setupSceneCulling(scene, cullingRoots, lightingSystem.sunLight.shadow);

  // Initialize static world matrices once, including the chunks and shadow proxies created above.
  // Crowd, tree and building-detail animation changes instance/pose buffers, not Object3D transforms.
  const dynamicRoots = [
    trafficSystem.group,
    highwayTraffic.group,
    scene.getObjectByName('StoreClusteredLights'),
    scene.getObjectByName('StreetClusteredLights'),
    ...Object.values(controllerManager.entities).map((entity) => entity.mesh),
    controllerManager.entities.cyclist.helmet
  ];
  scene.updateMatrixWorld(true);
  scene.matrixWorldAutoUpdate = false;
  markInit('culling');

  // 5. TSL RenderPipeline: geometry pre-pass, half-resolution stochastic SSR and SSGI resolved in the material BRDF,
  //    selectable FSR1 upscaling or native TRAA, Bloom & Filmic Toning
  await showLoadingStage('Setting up visual effects, map and interface...', 'ui');
  const camera = cameraSystem.camera;
  // ?bloom=0 in the address starts without the bloom (the Inspector switches it too). ?ssgiDenoise=0 leaves the
  // spatial denoiser out of SSGI: the materials are compiled against that chain, so the Inspector's switch for it
  // reloads the page.
  const address = new URLSearchParams(location.search);
  const rendering = { method: 'AMD FSR1 (75% resolution)', historySampling: 'none', bloom: address.get('bloom') !== '0', ssgiDenoise: address.get('ssgiDenoise') !== '0' };
  const fsrMethod = rendering.method;
  const nativeMethod = 'Native (100% resolution)';
  const sceneResolutionTargets = [];

  // The pre-pass draws the opaque geometry SSR and SSGI trace, without shading. It writes the lobe SSR answers: the
  // clearcoat where a material has one (the paint of the vehicles), the base lobe elsewhere
  const prePass = pass(scene, camera);
  const preMRT = mrt({
    output: opaqueFragment(packNormalToOctahedral(outermostNormalView)),
    metalrough: vec2(outermostMetalness, outermostRoughness),
    velocity: vec4(velocity, 0, previousViewZ)
  });
  // The attachments hold data, so the blending of transparent materials does not apply
  preMRT.setBlendMode('output', new THREE.BlendMode(THREE.NoBlending));
  prePass.setMRT(preMRT);

  const prePassNormal = prePass.getTextureNode('output');
  const prePassDepth = prePass.getTextureNode('depth');
  const prePassMetalRough = prePass.getTextureNode('metalrough');
  // The attachments clone the output texture, so the velocity is created before the output becomes RG
  const prePassVelocity = prePass.getTextureNode('velocity');
  const normalTexture = prePass.getTexture('output');
  normalTexture.format = THREE.RGFormat;
  normalTexture.minFilter = normalTexture.magFilter = THREE.NearestFilter; // octahedral normals can't be filtered
  prePass.getTexture('metalrough').type = THREE.UnsignedByteType;
  const normalDecode = (value) => unpackOctahedralToNormal(value.rg);

  // The beauty pass shades on top of the pre-pass's depth instead of clearing it, so the depth test rejects every
  // hidden fragment before it is shaded (no overdraw). TRAA jitters the camera for both passes alike.
  const beautyPass = pass(scene, camera, { depthTexture: prePass.renderTarget.depthTexture, autoClearDepth: false });

  // SSR reflects the previous frame's beauty, reprojected, so it is ready before this frame is shaded. The mip chain
  // serves the cone footprint of rough reflections.
  const previousBeauty = reprojectedBeauty(prePass, beautyPass);
  previousBeauty.renderTarget.texture.generateMipmaps = true;
  previousBeauty.renderTarget.texture.minFilter = THREE.LinearMipmapLinearFilter;

  // Reflections are traced and accumulated at half of the scene's resolution, from depth-aware downsampled inputs
  const downsample = depthAwareDownsample(prePassDepth, { normal: prePassNormal, metalrough: prePassMetalRough, velocity: prePassVelocity });
  const lowDepth = downsample.getTextureNode('depth');
  const lowNormal = downsample.getTextureNode('normal');
  const lowMetalRough = downsample.getTextureNode('metalrough');

  const lowSceneNormal = sample((uv) => normalDecode(lowNormal.sample(uv)));
  // Joint bilateral upsample to the scene's resolution, once per pixel rather than per shaded fragment. The tolerance
  // is the relative depth difference at which a texel is rejected: at UPSAMPLE_BILINEAR none is, which leaves the
  // plain bilinear weights (for comparison, in the inspector).
  const UPSAMPLE_JOINT = 0.005, UPSAMPLE_BILINEAR = 1e9;
  const upsampleTolerance = uniform(UPSAMPLE_JOINT);
  const upsample = (node, options) => {
    const target = rtt(jointBilateralUpsample(node, lowDepth, prePassDepth, camera, { depthTolerance: upsampleTolerance, ...options }).sample(screenUV));
    sceneResolutionTargets.push(target);
    return target.sample(screenUV);
  };

  const ssrNode = ssr(previousBeauty, lowDepth, lowSceneNormal, {
    camera,
    stochastic: true,
    metalnessNode: lowMetalRough.r,
    roughnessNode: lowMetalRough.g,
    // Hi-Z cone tracing requires WebGPU, the WebGL 2 fallback uses the linear tracer
    tracing: renderer.backend.isWebGPUBackend === true ? 'adaptive' : 'linear',
    coneTracing: true,
    rayReuse: true
  });
  ssrNode.adaptiveMaxSteps.value = 64;
  ssrNode.coneMaxLevel.value = 5;
  ssrNode.mirrorBias.value = 0.7;
  ssrNode.maxDistance.value = 150;
  ssrNode.thickness.value = 0;
  ssrNode.cameraRayThreshold.value = 1;

  // Temporal accumulation: ray reuse filters over space, so there is no spatial denoiser
  const ssrAccumulated = temporalReproject(ssrNode.getTextureNode(), lowDepth, lowNormal, downsample.getTextureNode('velocity'), camera, {
    mode: 'specular',
    hitDistance: ssrNode.getHitDistanceNode(),
    accumulate: true,
    normalDecode
  });
  ssrAccumulated.maxFrames.value = 16;
  ssrAccumulated.clampIntensity.value = 0.25;

  const reflection = upsample(ssrAccumulated.getTextureNode());

  // Screen-space global illumination at the same half resolution, gathered from the previous frame like SSR. One
  // pass returns both the light of the directions the screen occludes and how much of the hemisphere stays open
  // (its AO), so there is no separate AO pass.
  const giPass = ssgi(previousBeauty, lowDepth, lowSceneNormal, camera);
  giPass.sliceCount.value = 2;
  giPass.stepCount.value = 8;
  giPass.radius.value = 32;
  const giNode = giPass.getGINode();
  const aoNode = giPass.getAONode();
  // RGB is the bounced light, alpha the AO. A few samples per pixel are far too sparse on their own, so they are
  // filtered over space by an edge-aware blur (SVGF's à-trous wavelets, guided by the depth and the normals, two
  // passes for a span of 9 texels) and then accumulated over time like SSR, at half resolution, before the upsample.
  const giInput = rtt(vec4(giNode.sample(screenUV).rgb, aoNode.sample(screenUV).r), null, null, { resolutionScale: 0.5 });
  const giDenoised = rendering.ssgiDenoise ? denoise(giInput, { depth: lowDepth, normal: lowSceneNormal, camera, alpha: true }) : null;
  const giAccumulated = temporalReproject(giDenoised ? giDenoised.getTextureNode() : giInput, lowDepth, lowNormal, downsample.getTextureNode('velocity'), camera, {
    mode: 'diffuse',
    accumulate: true,
    normalDecode
  });
  const gi = upsample(giAccumulated.getTextureNode());

  // Sky occlusion: the share of the sky each pixel sees over the skyline around it, from the height map of the city
  // (js/sky-occlusion.js), at half resolution like SSGI. The part of the sky the buildings hide shows the buildings
  // instead: it is filled with the light they bounce, which is the ground colour of the time of day's hemisphere
  // light scaled by its canyon bounce (set in the loop), the same from every direction.
  // The position is rebuilt from the depth, which gets coarse far away: the height bias grows with the distance.
  const cameraWorld = uniform(camera.matrixWorld);
  const lowViewPosition = getViewPosition(screenUV, lowDepth.sample(screenUV).r, uniform(camera.projectionMatrixInverse));
  const skyVisibilityLow = rtt(vec4(skyOcclusion.visibility(
    cameraWorld.mul(vec4(lowViewPosition, 1)).xyz,
    cameraWorld.mul(vec4(normalDecode(lowNormal.sample(screenUV)), 0)).xyz.normalize(),
    lowViewPosition.z.negate().mul(0.002)
  )), null, null, { resolutionScale: 0.5 });
  const skyVisibility = upsample(skyVisibilityLow).r;
  const skyIntensity = uniform(1); // 0: the canyon AO of the materials instead (the look before the height map)
  const skyBounce = uniform(new THREE.Color()); // the irradiance of a surface that sees only buildings
  const skyBounceGain = uniform(1); // 0: black
  const skyCoverage = skyVisibility.oneMinus().mul(skyIntensity);
  // Before every render of the main view (the hero probe's renders switch the canyon AO back on): the sky occlusion
  // replaces the canyon AO of the materials, and the bounce follows the time of day
  function syncSkyOcclusion() {
    cityUniforms.uCanyonAO.value = 1 - skyIntensity.value;
    const hemiLight = lightingSystem.hemiLight;
    skyBounce.value.copy(hemiLight.groundColor).multiplyScalar(hemiLight.intensity * cityUniforms.uCanyonBounce.value);
  }

  // Hero reflection probe: a cube map of the surroundings captured at the player's vehicle, without the vehicle. It
  // answers what the screen can't (the street behind the camera, in the paintwork) for the vehicle alone, where it
  // stands in for the sky-only environment. The road and the cars next to it are not at the probe's centre.
  const HERO_SIZE = 128;
  const heroMips = Math.log2(HERO_SIZE);
  const heroTarget = new THREE.CubeRenderTarget(HERO_SIZE, { type: THREE.HalfFloatType, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter });
  const heroCamera = new THREE.CubeCamera(0.3, 200000, heroTarget); // as far as the sky dome
  const heroIntensity = uniform(1); // 0 switches the probe off (it is then not updated either)
  let heroMesh = null; // the player's vehicle (or person), whose meshes the probe lights
  const isHero = uniform(0).onObjectUpdate(({ object }) => {
    for (let o = object; o !== null; o = o.parent) if (o === heroMesh) return 1;
    return 0;
  });
  const heroCoverage = isHero.mul(heroIntensity);
  // ponytail: the cube's plain mip chain stands in for a GGX prefilter (PMREM), and its 2x2 level for the irradiance;
  // prefilter it if rough or diffuse surfaces near the vehicle look wrong
  const heroSpecular = ({ direction, roughness }) => vec4(cubeTexture(heroTarget.texture, direction).level(roughness.mul(heroMips)).rgb.mul(heroCoverage), heroCoverage);
  const heroDiffuse = ({ normal }) => vec4(cubeTexture(heroTarget.texture, normal).level(heroMips - 1).rgb.mul(Math.PI).mul(heroCoverage), heroCoverage);
  let heroFace = -1;
  // One face per frame (a face is one more draw of the city), after the frame is rendered. All six on the first call.
  // The probe is lit with the sun's shadow maps of the frame: three.js would fit and draw the cascades again for the
  // camera of the face (two more draws of the city's casters). The frame's cascades are fitted to the view, so the
  // probe's shadows are right around the vehicle and in front of the camera, and not to be trusted behind it.
  function updateHeroProbe() {
    const entity = controllerManager.getActiveEntity();
    heroMesh = entity?.mesh ?? null;
    if (heroMesh === null || heroIntensity.value === 0) return;
    heroCamera.position.copy(entity.position).y += 1;
    heroCamera.updateMatrixWorld();
    const visible = entity.mesh.visible;
    entity.mesh.visible = false;
    cityUniforms.uCanyonAO.value = 1; // no sky occlusion pass here (the loop sets it back for the main view)
    const sunShadow = lightingSystem.sunLight.shadow;
    const autoUpdate = sunShadow.autoUpdate;
    sunShadow.autoUpdate = false;
    if (heroFace < 0) {
      heroCamera.update(renderer, scene);
    } else {
      renderer.setRenderTarget(heroTarget, heroFace);
      renderer.render(scene, heroCamera.children[heroFace]);
      renderer.setRenderTarget(null);
    }
    heroFace = (heroFace + 1) % 6;
    entity.mesh.visible = visible;
    sunShadow.autoUpdate = autoUpdate;
  }

  // The materials read all of it through their BRDF, each source filling what the one before left uncovered.
  // Indirect specular: SSR, the hero probe, the environment. Indirect diffuse: SSGI (whose coverage is its occlusion,
  // so the AO is not applied a second time), the hero probe, the environment.
  // For the inspector's two indirect light outputs, a last specular source that answers nothing notes the lighting
  // variables of the shader being built: its radiance and irradiance are what those lists and the environment resolve to.
  const lightingOf = new WeakMap(); // by the builder's node list: Fn() gets a proxy of the builder, not the builder
  const noteLighting = (query, builder) => (lightingOf.set(builder.nodes, builder.context), null);
  const resolvedLight = (get) => Fn((builder) => lightingOf.has(builder.nodes) ? vec4(get(lightingOf.get(builder.nodes)), 1)
    : vec4(0, 0, 0, builder.material.transparent ? 0 : 1))(); // unlit: black, or what is behind it if transparent
  const indirectMRT = mrt({
    output,
    indirectSpecular: resolvedLight((c) => { // of the lobe SSR answers: the clearcoat where there is one
      const coat = c.lightingModel.clearcoatRadiance;
      return coat ? clearcoat.greaterThan(0).select(coat, c.radiance) : c.radiance;
    }),
    // before the material's own AO. The hemisphere light is in it: it fills what the sources leave open, like the environment
    indirectDiffuse: resolvedLight((c) => c.irradiance.add(c.occlusionAwareIrradiance))
  });
  // The alpha above decides what a fragment replaces, whatever the blending of its material
  indirectMRT.setBlendMode('indirectSpecular', new THREE.BlendMode(THREE.NormalBlending));
  indirectMRT.setBlendMode('indirectDiffuse', new THREE.BlendMode(THREE.NormalBlending));
  // An intensity scales a source's light and coverage together, so the sources behind it fill what it gives up
  // (at 0 the hero probe and the environment answer everything, and SSGI's AO is gone with its light)
  const ssrIntensity = uniform(1);
  const giIntensity = uniform(1);
  beautyPass.contextNode = builtinIndirectContext({
    specular: [opaqueFragmentSource(ssrNode.getSpecularSource(reflection).mul(ssrIntensity)), heroSpecular, noteLighting],
    // the sky occlusion covers the part of the environment the buildings hide, with the light they bounce
    diffuse: [opaqueFragmentSource(giPass.getDiffuseSource(gi.rgb, gi.a).mul(giIntensity)), heroDiffuse,
      opaqueFragmentSource(vec4(skyBounce.mul(skyBounceGain).mul(skyCoverage), skyCoverage))]
  });

  // TRAA is off by default and can be switched on in the Inspector, at scene resolution in both modes (FSR1 is
  // meant to upscale an anti-aliased source).
  // History sampling is compiled into TRAA's resolve shader. Keep a separate graph per filter;
  // only the selected graph renders, and its history is reset whenever the selection changes.
  // 'none' is the beauty pass as it is, without TRAA (and so without its camera jitter), for comparison: FSR1 then
  // upscales the aliased edges, and the noise SSR and SSGI leave is no longer averaged out.
  const variants = Object.fromEntries(['catmull-rom', 'linear', 'none'].map((filter) => {
    const antialiased = filter === 'none' ? null : traa(beautyPass, prePassDepth, prePassVelocity, camera);
    if (antialiased) antialiased.historySampling = filter;
    const source = antialiased ? antialiased.getTextureNode() : beautyPass.getTextureNode('output');
    const upscaled = fsr1(source).getTextureNode();
    // TRAA builds the pre-pass before the beauty pass, which shades on top of its depth. Without TRAA nothing in the
    // output graph does, and the beauty pass would render first, on last frame's depth (of another size after a
    // change of resolution): a read of the depth that adds nothing puts the pre-pass first.
    const afterPrePass = (node) => antialiased ? node : prePassDepth.sample(screenUV).r.mul(0).add(node);
    const sceneColors = { [fsrMethod]: afterPrePass(upscaled), [nativeMethod]: antialiased ?? afterPrePass(source) };
    // the bloom is made from the scene before FSR1 upscales it
    const grades = Object.fromEntries(Object.entries(sceneColors).map(([method, node]) => [method,
      method === fsrMethod ? createPostProcessing(renderer, node, source) : createPostProcessing(renderer, node)]));
    return [filter, { antialiased, sceneColors, grades }];
  }));
  const selectedVariant = () => variants[rendering.historySampling];
  const selectedSceneColor = () => selectedVariant().sceneColors[rendering.method];
  const allGrades = Object.values(variants).flatMap((variant) => Object.values(variant.grades));
  // Keep time-of-day grading in sync across the inactive graphs too.
  const colorGrade = {
    get outputNode() { return selectedVariant().grades[rendering.method][rendering.bloom ? 'outputNode' : 'outputNodeWithoutBloom']; },
    applyPreset: (...args) => allGrades.forEach((grade) => grade.applyPreset(...args)),
    transitionTo: (...args) => allGrades.forEach((grade) => grade.transitionTo(...args)),
    update: () => allGrades.forEach((grade) => grade.update())
  };
  function setRenderingResolution() {
    const scale = rendering.method === fsrMethod ? 0.75 : 1;
    for (const target of [prePass, beautyPass, previousBeauty, ...sceneResolutionTargets]) target.setResolutionScale(scale);
    giInput.setResolutionScale(0.5 * scale);
    skyVisibilityLow.setResolutionScale(0.5 * scale);
    allGrades.forEach((grade) => grade.setResolutionScale(scale));
    selectedVariant().antialiased?.reset(); // scene and temporal histories resize on their next render
  }
  setRenderingResolution();
  const renderPipeline = new THREE.RenderPipeline(renderer, colorGrade.outputNode);

  // ?inspector=true in the address shows the three.js Inspector: its Parameters tab (the pipeline's buffers as
  // outputs and the rendering, SSR and SSGI settings) and its Performance tab (the frame rate and the CPU / GPU time
  // of each pass). It is not loaded otherwise.
  let inspector = null;
  if (address.get('inspector') === 'true') import('three/addons/inspector/Inspector.js').then(({ Inspector }) => {
    inspector = new Inspector();
    renderer.inspector = inspector;
    inspector.domElement.style.zIndex = 1000; // above the HUD
    addInspectorParameters();
  });
  // profilePasses(seconds) in the console, or tools/profile-passes.mjs: where the time of a frame goes, by pass
  // (js/pass-profiler.js). The scene's passes are named by what they draw to.
  // It is there once the city is on screen.
  const profilePasses = async (seconds = 5) => {
    const { PassProfiler, passLabel, formatPassProfile } = await import('./pass-profiler.js');
    const targets = new Map([[prePass.renderTarget, 'Pre-pass'], [beautyPass.renderTarget, 'Beauty pass'], [heroTarget, 'Hero probe']]);
    const profiler = new PassProfiler({ label: (stats) => targets.get(stats.renderTarget) ?? passLabel(stats) });
    const previous = renderer.inspector;
    renderer.inspector = profiler;
    const report = await profiler.measure(seconds);
    renderer.inspector = inspector ?? previous; // the Inspector, if it was opened meanwhile
    report.text = formatPassProfile(report);
    console.log(report.text);
    return report;
  };
  const show = (node) => vec4(vec3(node), 1);
  // The inspector's two indirect light outputs, set up when first asked for: the beauty pass writes them to
  // attachments of their own, which recompiles its shaders (a pause).
  // ponytail: the attachments stay until the page is reloaded; drop them on leaving if their cost gets in the way
  let indirectOutputs = null;
  function getIndirectOutputs() {
    if (indirectOutputs === null) {
      beautyPass.setMRT(indirectMRT);
      indirectOutputs = {
        'Indirect Specular': show(beautyPass.getTextureNode('indirectSpecular').rgb),
        'Indirect Diffuse': show(beautyPass.getTextureNode('indirectDiffuse').rgb.div(Math.PI)) // as a white surface reflects it
      };
    }
    return indirectOutputs;
  }
  function addInspectorParameters() {
    const viewZ = getViewPosition(screenUV, prePassDepth.sample(screenUV).r, uniform(camera.projectionMatrixInverse)).z;
    // The world position and normal of the pixel, for the sky occlusion outputs (the sky itself shows as open)
    const viewPosition = getViewPosition(screenUV, prePassDepth.sample(screenUV).r, uniform(camera.projectionMatrixInverse));
    const worldPosition = cameraWorld.mul(vec4(viewPosition, 1)).xyz;
    const worldNormal = cameraWorld.mul(vec4(normalDecode(prePassNormal.sample(screenUV)), 0)).xyz.normalize();
    const isSky = viewZ.lessThan(-50000);
    const outputs = {
      'Final': colorGrade.outputNode,
      'Beauty (before anti-aliasing)': show(beautyPass.getTextureNode('output').rgb),
      'Beauty (after reconstruction)': show(selectedSceneColor().rgb),
      'Reprojected previous beauty': show(previousBeauty.sample(screenUV).rgb),
      'Reprojection validity': show(previousBeauty.sample(screenUV).a),
      'Depth (view distance / 200 m)': show(viewZ.negate().div(200)),
      'Normal': show(normalDecode(prePassNormal.sample(screenUV)).mul(0.5).add(0.5)),
      'Metalness': show(prePassMetalRough.sample(screenUV).r),
      'Roughness': show(prePassMetalRough.sample(screenUV).g),
      'Velocity (x100)': show(prePassVelocity.sample(screenUV).xy.abs().mul(100)),
      'SSR traced (half res)': show(ssrNode.getTextureNode().sample(screenUV).rgb),
      'SSR accumulated': show(reflection.rgb),
      'SSR coverage': show(reflection.a),
      'SSGI light (half res)': show(giNode.sample(screenUV).rgb),
      ...(giDenoised ? { 'SSGI light blurred (half res)': show(giDenoised.getTextureNode().sample(screenUV).rgb) } : {}),
      'SSGI light accumulated': show(gi.rgb),
      'SSGI AO': show(gi.a),
      'SSGI coverage': show(gi.a.oneMinus()),
      'Sky visibility (in use: half res, upsampled)': show(skyVisibility),
      'Sky visibility (height map)': show(isSky.select(1, skyOcclusion.visibility(worldPosition, worldNormal))),
      'Sky visibility (brute force reference, slow)': show(isSky.select(1, skyOcclusion.visibilityReference(worldPosition, worldNormal))),
      'Canyon AO (height only)': show(isSky.select(1, canyonSkyAO(worldPosition))),
      'Height map (1 km around the camera, white = 300 m)': show(skyOcclusion.topDown(screenUV, screenSize.x.div(screenSize.y), cameraWorld.mul(vec4(0, 0, 0, 1)))),
      'Indirect Specular': null, // these two are set when first shown, see below
      'Indirect Diffuse': null,
      // equirectangular: up is +y, the centre of the image looks along -z (north)
      'Hero probe': show(cubeTexture(heroTarget.texture, vec3(screenUV.x.mul(Math.PI * 2).sin().mul(screenUV.y.mul(Math.PI).sin()), screenUV.y.mul(Math.PI).cos(), screenUV.x.mul(Math.PI * 2).cos().mul(screenUV.y.mul(Math.PI).sin()))).level(0).rgb)
    };
    const state = { output: 'Final' };
    const showOutput = (name) => {
      if (outputs[name] === null) Object.assign(outputs, getIndirectOutputs());
      renderPipeline.outputNode = outputs[name];
      renderPipeline.needsUpdate = true;
    };
    const activeRendering = { ...rendering };
    const updateRendering = (property, control) => {
      if (capturing) { // the capture uses a fixed graph and resolution until it finishes
        if (rendering[property] !== activeRendering[property]) control.setValue(activeRendering[property]);
        return;
      }
      Object.assign(activeRendering, rendering);
      setRenderingResolution();
      outputs.Final = colorGrade.outputNode;
      outputs['Beauty (after reconstruction)'] = show(selectedSceneColor().rgb);
      showOutput(state.output);
    };
    const renderingGui = inspector.createParameters('Rendering');
    const methodControl = renderingGui.add(rendering, 'method', [fsrMethod, nativeMethod]).name('Method')
      .onChange(() => updateRendering('method', methodControl));
    const samplingControl = renderingGui.add(rendering, 'historySampling', { 'Catmull-Rom history': 'catmull-rom', 'Bilinear history': 'linear', 'Off': 'none' }).name('TRAA')
      .onChange(() => updateRendering('historySampling', samplingControl));
    const bloomControl = renderingGui.add(rendering, 'bloom').name('Bloom').onChange(() => updateRendering('bloom', bloomControl));
    window.rendering = { set: (property, value) => ({ historySampling: samplingControl, bloom: bloomControl, method: methodControl })[property].setValue(value) }; // for scripted screenshots
    renderingGui.add(upsampleTolerance, 'value', { 'Joint bilateral': UPSAMPLE_JOINT, 'Bilinear': UPSAMPLE_BILINEAR }).name('SSR / SSGI upsampling');
    const outputControl = inspector.createParameters('Output').add(state, 'output', Object.keys(outputs)).name('Show').onChange(showOutput);
    window.showOutput = (name) => outputControl.setValue(name); // for scripted screenshots
    const ssrGui = inspector.createParameters('SSR');
    ssrGui.add(ssrIntensity, 'value', 0, 1, 0.01).name('Intensity');
    ssrGui.add(ssrNode.maxDistance, 'value', 1, 500, 1).name('Max distance (m)');
    ssrGui.add(ssrNode.thickness, 'value', 0, 2, 0.01).name('Thickness');
    ssrGui.add(ssrNode.mirrorBias, 'value', 0, 1, 0.01).name('Mirror bias');
    ssrGui.add(ssrNode.cameraRayThreshold, 'value', 0, 1, 0.01).name('Camera ray threshold');
    ssrGui.add(ssrNode.adaptiveMaxSteps, 'value', 16, 256, 16).name('Max steps');
    ssrGui.add(ssrNode.coneMaxLevel, 'value', 0, 8, 1).name('Cone max mip');
    ssrGui.add(ssrNode, 'rayReuse').name('Ray reuse');
    ssrGui.add(ssrNode, 'coneTracing').name('Cone tracing');
    ssrGui.add(ssrAccumulated.maxFrames, 'value', 1, 64, 1).name('History frames');
    ssrGui.add(ssrAccumulated.clampIntensity, 'value', 0, 1, 0.01).name('History clamp');
    const giGui = inspector.createParameters('SSGI');
    giGui.add(giIntensity, 'value', 0, 1, 0.01).name('Intensity');
    giGui.add(giPass.giIntensity, 'value', 0, 30, 0.1).name('Bounce gain');
    giGui.add(giPass.aoIntensity, 'value', 0, 4, 0.01).name('AO intensity');
    giGui.add(giPass.sliceCount, 'value', 1, 4, 1).name('Slices');
    giGui.add(giPass.stepCount, 'value', 1, 32, 1).name('Steps');
    giGui.add(giPass.radius, 'value', 1, 50, 0.1).name('Radius');
    giGui.add(giPass.expFactor, 'value', 1, 3, 0.01).name('Exp factor');
    giGui.add(giPass.thickness, 'value', 0.01, 10, 0.01).name('Thickness');
    giGui.add(giPass.backfaceLighting, 'value', 0, 1, 0.01).name('Backface lighting');
    giGui.add(giPass.useLinearThickness, 'value').name('Linear thickness');
    giGui.add(giPass.useScreenSpaceSampling, 'value').name('Screen-space sampling');
    giGui.add(giPass, 'useTemporalFiltering').name('Per-frame noise');
    giGui.add(rendering, 'ssgiDenoise').name('Spatial denoise (reloads)').onChange((on) => {
      if (on) address.delete('ssgiDenoise');
      else address.set('ssgiDenoise', '0');
      location.search = address.toString();
    });
    if (giDenoised) giGui.add(giDenoised.luminancePhi, 'value', 0, 10, 0.1).name('Blur strength');
    giGui.add(giAccumulated.maxFrames, 'value', 1, 64, 1).name('History frames');
    const skyGui = inspector.createParameters('Sky occlusion');
    skyGui.add(skyIntensity, 'value', { 'Height map': 1, 'Canyon AO (before)': 0 }).name('From');
    skyGui.add(skyBounceGain, 'value', 0, 3, 0.01).name('Bounce (0 = black)');
    Object.assign(window, { skyIntensity, skyBounceGain }); // for scripted screenshots
    const heroGui = inspector.createParameters('Hero probe');
    heroGui.add(heroIntensity, 'value', 0, 1, 0.01).name('Coverage (0 = off)');
  }

  // The capture button saves the lighting breakdown of the current view as an animated WebP: the beauty, its two
  // indirect lights, then the beauty again with the hero probe, SSGI, SSR and the sky occlusion of the height map
  // switched off one after the other. The
  // world stands still meanwhile (see the loop), each frame is rendered at 1920 x 1080 and labelled, and the file is
  // assembled in the page (no server).
  const CAPTURE_WIDTH = 1920, CAPTURE_HEIGHT = 1080;
  const CAPTURE_SETTLE_FRAMES = 72; // for TRAA and the SSR / SSGI histories (up to 64 frames) to converge
  const CAPTURE_FRAME_MS = 1500;
  let capturing = false;
  let afterCaptureFrame = null; // called by the loop once the frame is rendered, while the canvas still holds it
  const captureFrame = (grab) => new Promise((resolve) => {
    afterCaptureFrame = () => { afterCaptureFrame = null; resolve(grab?.()); };
  });
  const captureButton = document.getElementById('btn-capture');
  if (captureButton) captureButton.addEventListener('click', async () => {
    if (capturing) return;
    const showOutput = (node) => { renderPipeline.outputNode = node; renderPipeline.needsUpdate = true; };
    const steps = [
      ['Beauty', () => showOutput(colorGrade.outputNode)],
      ['Indirect specular', () => showOutput(getIndirectOutputs()['Indirect Specular'])],
      ['Indirect diffuse', () => showOutput(getIndirectOutputs()['Indirect Diffuse'])],
      ['Beauty', () => showOutput(colorGrade.outputNode)],
      ['Hero probe off', () => { heroIntensity.value = 0; }],
      ['Hero probe and SSGI off', () => { giIntensity.value = 0; }],
      ['Hero probe, SSGI and SSR off', () => { ssrIntensity.value = 0; }],
      // the materials' canyon AO takes its place, as before the height map
      ['Hero probe, SSGI, SSR and sky occlusion height map off', () => { skyIntensity.value = 0; }]
    ];
    const before = { output: renderPipeline.outputNode, hero: heroIntensity.value, gi: giIntensity.value, ssr: ssrIntensity.value, sky: skyIntensity.value };
    const title = captureButton.title;
    const canvas = document.createElement('canvas');
    canvas.width = CAPTURE_WIDTH;
    canvas.height = CAPTURE_HEIGHT;
    const ctx = canvas.getContext('2d', { alpha: false });
    capturing = true;
    captureButton.disabled = true;
    try {
      renderer.setSize(CAPTURE_WIDTH, CAPTURE_HEIGHT, false); // the drawing buffer only: the page keeps its layout
      cameraSystem.onResize(CAPTURE_WIDTH, CAPTURE_HEIGHT);
      const stills = [];
      for (const [i, [label, apply]] of steps.entries()) {
        captureButton.title = `Capturing ${i + 1} / ${steps.length}: ${label}`;
        apply();
        for (let f = 0; f < CAPTURE_SETTLE_FRAMES; f++) await captureFrame();
        await captureFrame(() => ctx.drawImage(renderer.domElement, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT));
        ctx.font = '600 40px system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(40, CAPTURE_HEIGHT - 120, ctx.measureText(label).width + 48, 80);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, 64, CAPTURE_HEIGHT - 80);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/webp', 0.92));
        stills.push(new Uint8Array(await blob.arrayBuffer()));
      }
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([animatedWebP(stills, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FRAME_MS)], { type: 'image/webp' }));
      link.download = 'slop-newyork-lighting-breakdown.webp';
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 60000);
    } finally {
      heroIntensity.value = before.hero;
      giIntensity.value = before.gi;
      ssrIntensity.value = before.ssr;
      skyIntensity.value = before.sky;
      showOutput(before.output);
      renderer.setSize(window.innerWidth, window.innerHeight);
      cameraSystem.onResize(window.innerWidth, window.innerHeight);
      capturing = false;
      captureButton.disabled = false;
      captureButton.title = title;
    }
  });

  // For scripted screenshots (tools outside the page): hold() stops the world as the capture does and sets the
  // drawing buffer's size, grab() resolves with a rendered frame as a PNG data URL after `settle` frames, and the
  // switches of the sources are there to set. grab() also works while the world runs (for the motion vectors).
  window.shots = {
    heroIntensity, giIntensity, ssrIntensity, skyIntensity, skyBounceGain, rendering,
    hold(on, width = CAPTURE_WIDTH, height = CAPTURE_HEIGHT) {
      capturing = on;
      renderer.setSize(on ? width : window.innerWidth, on ? height : window.innerHeight, !on);
      cameraSystem.onResize(on ? width : window.innerWidth, on ? height : window.innerHeight);
    },
    async grab(settle = CAPTURE_SETTLE_FRAMES) {
      for (let f = 0; f < settle; f++) await captureFrame();
      return captureFrame(() => renderer.domElement.toDataURL('image/png'));
    }
  };

  // 6. Wire UI Controls (Strictly using safe DOM APIs: textContent, classList, addEventListener)
  const modeButtons = document.querySelectorAll('[data-mode]');
  const cameraButtons = document.querySelectorAll('[data-camera]');
  const timeButtons = document.querySelectorAll('[data-time]');
  const teleportButtons = document.querySelectorAll('[data-teleport]');
  const hudToggleBtn = document.getElementById('btn-toggle-hud');
  const modeSwitch = document.getElementById('mode-switch');
  touchControls.onAvailabilityChange = placeModeIndicator;

  const elFps = document.getElementById('hud-fps');
  const elMode = document.getElementById('hud-active-mode');
  const elCamera = document.getElementById('hud-active-camera');
  const elSpeed = document.getElementById('hud-speed');
  const elSpeedMeter = document.getElementById('hud-speed-meter');
  const elAltitude = document.getElementById('hud-altitude');
  const elLocation = document.getElementById('hud-location');
  const elControlsHint = document.getElementById('hud-controls-hint');
  const gamepadControls = new GamepadControls(controllerManager, cameraSystem, {
    onCycleMode: (step) => {
      const modes = ['person', 'car', 'cyclist', 'helicopter'];
      selectMode(modes[(modes.indexOf(controllerManager.activeMode) + step + modes.length) % modes.length]);
    },
    onCycleCamera: cycleCamera,
    onToggleMap: () => mapSystem.toggleFullMap(!mapSystem.isFullMapOpen),
    onCloseMap: () => mapSystem.toggleFullMap(false)
  });

  // Keyboard guide for each mode: [keys, action]
  const controlHints = {
    person: [[['W', 'S'], 'Walk / Run'], [['A', 'D'], 'Turn'], [['Space'], 'Jump'], [['Drag'], 'Orbit']],
    car: [[['W', 'S'], 'Drive / Reverse'], [['A', 'D'], 'Steer'], [['B'], 'Horn'], [['Drag'], 'Orbit']],
    cyclist: [[['W', 'S'], 'Pedal / Brake'], [['Space'], 'Brake'], [['A', 'D'], 'Lean & Turn'], [['Drag'], 'Orbit']],
    helicopter: [[['W', 'S'], 'Pitch'], [['A', 'D'], 'Yaw'], [['Space', 'E'], 'Ascend'], [['Q', 'C'], 'Descend']]
  };
  // Top speeds in mph (controllers.js caps them at 8.2, 36, 19 & 52 m/s): full scale of the speed meter
  const topSpeedMph = { person: 18, car: 81, cyclist: 43, helicopter: 116 };

  function renderControlHints(mode) {
    if (!elControlsHint) return;
    const items = (controlHints[mode] || []).map(([keys, action]) => {
      const item = document.createElement('span');
      item.className = 'hint';
      for (const key of keys) {
        const kbd = document.createElement('kbd');
        kbd.textContent = key;
        item.appendChild(kbd);
      }
      const label = document.createElement('span');
      label.className = 'hint-label';
      label.textContent = action;
      item.appendChild(label);
      return item;
    });
    elControlsHint.replaceChildren(...items);
  }

  // The taxi-yellow pill under the selected mode glides from button to button
  function placeModeIndicator() {
    const active = modeSwitch ? modeSwitch.querySelector('.mode-btn.active') : null;
    if (!active) return;
    modeSwitch.style.setProperty('--ind-x', `${active.offsetLeft}px`);
    modeSwitch.style.setProperty('--ind-w', `${active.offsetWidth}px`);
  }

  function selectMode(mode) {
    controllerManager.setMode(mode);
    touchControls.setMode(controllerManager.activeMode);
    gamepadControls.reset();
    modeButtons.forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    });
    document.body.dataset.mode = mode;
    placeModeIndicator();
    const ent = controllerManager.getActiveEntity();
    if (elMode) elMode.textContent = ent.label;
    renderControlHints(mode);

    // If currently on a static scenic camera, switch to 3rd-Person Chase so user sees their selected character/vehicle
    const currentCam = cameraSystem.getActivePreset();
    if (currentCam.category !== 'follow') {
      selectCamera('chase', false);
    }
  }

  function selectCamera(camId, snap = false) {
    cameraSystem.setCameraPreset(camId, snap);
    cameraButtons.forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-camera') === camId);
    });
    const preset = cameraSystem.getActivePreset();
    if (elCamera) elCamera.textContent = preset.name;
  }

  // Steps through CAMERA_PRESETS (V / Shift+V, and the arrows either side of the camera name)
  function cycleCamera(step) {
    const n = CAMERA_PRESETS.length;
    const idx = CAMERA_PRESETS.findIndex((p) => p.id === cameraSystem.activePresetId);
    selectCamera(CAMERA_PRESETS[(idx + step + n) % n].id, true);
  }

  // Changes of time of day blend over a couple of seconds (`instant` for the first one)
  const TIME_TRANSITION = 2;
  function selectTime(timeId, instant = false) {
    if (instant) {
      lightingSystem.applyPreset(timeId);
      colorGrade.applyPreset(lightingSystem.activePresetId);
    } else if (timeId !== lightingSystem.activePresetId) {
      lightingSystem.transitionTo(timeId, TIME_TRANSITION);
      colorGrade.transitionTo(lightingSystem.activePresetId, TIME_TRANSITION);
    }
    timeButtons.forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-time') === timeId);
    });
  }

  // Popovers (native [popover]: light dismiss & Esc for free) open against the control that toggles them: data-anchor
  // names the element they sit above / below, data-align the one whose right edge they line up with
  const popovers = document.querySelectorAll('.popover[popover]');
  const supportsPopover = typeof HTMLElement.prototype.togglePopover === 'function';
  function closePopovers() {
    if (!supportsPopover) return;
    popovers.forEach((pop) => {
      if (pop.matches(':popover-open')) pop.hidePopover();
    });
  }
  popovers.forEach((pop) => {
    const invoker = document.querySelector(`[popovertarget="${pop.id}"]`);
    pop.addEventListener('beforetoggle', (e) => {
      const open = e.newState === 'open';
      if (invoker) invoker.classList.toggle('is-open', open);
      if (!open) return;
      const anchor = document.getElementById(pop.dataset.anchor) || invoker;
      const align = document.getElementById(pop.dataset.align) || invoker;
      if (!anchor || !align) return;
      const r = anchor.getBoundingClientRect();
      const spaceAbove = Math.max(0, r.top - 20);
      const spaceBelow = Math.max(0, window.innerHeight - r.bottom - 20);
      let above = pop.dataset.side === 'above';
      // The camera control moves to the top on touch screens; open toward the available space.
      if ((above ? spaceAbove : spaceBelow) < 160) above = spaceAbove > spaceBelow;
      pop.style.right = `${Math.max(8, window.innerWidth - align.getBoundingClientRect().right)}px`;
      if (above) {
        pop.style.top = 'auto';
        pop.style.bottom = `${window.innerHeight - r.top + 8}px`;
        pop.style.maxHeight = `${spaceAbove}px`;
      } else {
        pop.style.bottom = 'auto';
        pop.style.top = `${r.bottom + 8}px`;
        pop.style.maxHeight = `${spaceBelow}px`;
      }
    });
  });

  modeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      selectMode(btn.getAttribute('data-mode'));
    });
  });

  cameraButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      selectCamera(btn.getAttribute('data-camera'), true);
      closePopovers();
    });
  });

  const camPrevBtn = document.getElementById('cam-prev');
  const camNextBtn = document.getElementById('cam-next');
  if (camPrevBtn) camPrevBtn.addEventListener('click', () => cycleCamera(-1));
  if (camNextBtn) camNextBtn.addEventListener('click', () => cycleCamera(1));

  timeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      selectTime(btn.getAttribute('data-time'));
    });
  });

  // A mouse click mustn't leave the focus on a HUD button, or Space (jump / ascend) and Enter would press it again
  document.addEventListener('click', (e) => {
    const btn = e.target instanceof Element ? e.target.closest('button') : null;
    if (btn && e.detail > 0) btn.blur();
  });

  // Where you are: the place's name over its street corner, drawn as a pair of NYC street-name signs
  const roadNames = new Set(ROADS.map((r) => r.name));
  const placeNames = new Set(NYC_LANDMARKS.map((l) => l.name));
  let locationText = '';
  function renderLocation(label) {
    if (!elLocation || label === locationText) return;
    locationText = label;
    let place = '';
    let streets = label;
    const split = label.indexOf(' · ');
    if (split >= 0) {
      place = label.slice(0, split);
      streets = label.slice(split + 3);
    } else if (placeNames.has(label) || (!label.includes(' & ') && !roadNames.has(label))) {
      place = label;
      streets = '';
    }
    const nodes = [];
    if (place) {
      const el = document.createElement('div');
      el.className = 'location-place';
      el.textContent = place;
      nodes.push(el);
    }
    if (streets) {
      const row = document.createElement('div');
      row.className = 'location-signs';
      if (streets.startsWith('near ')) {
        streets = streets.slice(5);
        const near = document.createElement('span');
        near.className = 'location-near';
        near.textContent = 'near';
        row.appendChild(near);
      }
      for (const name of streets.split(' & ')) {
        const sign = document.createElement('span');
        sign.className = 'street-sign';
        sign.textContent = name;
        row.appendChild(sign);
      }
      nodes.push(row);
    }
    elLocation.replaceChildren(...nodes);
  }

  function teleportTo(x, z, heading = Math.PI) {
    const destination = controllerManager.teleportActiveTo(x, z, heading);
    touchControls.reset();
    gamepadControls.reset();
    cameraSystem.resetForTeleport();
    selectCamera('chase', true);
    mapSystem.waypoint = {
      x: destination.x,
      z: destination.z,
      name: getLocationLabel(destination.x, destination.z)
    };
    return destination;
  }

  const mapSystem = new NYCMapSystem({
    controllerManager,
    cameraSystem,
    allBlocks: cityBlocks.allBlocks,
    onToggle: (open) => {
      controllerManager.resetInput();
      touchControls.setEnabled(!open);
      gamepadControls.setEnabled(!open);
      const mapHint = document.getElementById('fullmap-hover-info');
      if (open && touchControls.available && mapHint) {
        mapHint.textContent = 'Tap to teleport · Drag to pan · Use + / − to zoom';
      }
    },
    onTeleport: teleportTo
  });

  teleportButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-teleport');
      const loc = TELEPORT_LOCATIONS[key];
      if (loc) {
        teleportTo(loc.x, loc.z, loc.heading);
      }
      closePopovers();
    });
  });

  const openMapBtn = document.getElementById('btn-open-map');
  if (openMapBtn) openMapBtn.addEventListener('click', () => mapSystem.toggleFullMap(true));

  if (hudToggleBtn) {
    hudToggleBtn.addEventListener('click', () => {
      document.body.classList.toggle('hud-minimized');
      const isMin = document.body.classList.contains('hud-minimized');
      hudToggleBtn.setAttribute('aria-pressed', String(isMin));
      hudToggleBtn.setAttribute('aria-label', isMin ? 'Show HUD' : 'Hide HUD');
      hudToggleBtn.title = isMin ? 'Show the HUD [H]' : 'Clean view: hide the HUD [H]';
      if (isMin) closePopovers();
    });
  }

  // Global keyboard shortcuts for modes (1-4), cameras (V / 5-0), and HUD (H)
  window.addEventListener('keydown', (e) => {
    if (isTypingInField(e)) return;
    switch (e.code) {
      case 'Digit1':
        selectMode('person');
        break;
      case 'Digit2':
        selectMode('car');
        break;
      case 'Digit3':
        selectMode('cyclist');
        break;
      case 'Digit4':
        selectMode('helicopter');
        break;
      case 'Digit5':
        selectCamera('chase', true);
        break;
      case 'Digit6':
        selectCamera('action', true);
        break;
      case 'Digit7':
        selectCamera('pov', true);
        break;
      case 'Digit8':
        selectCamera('fifthAve', true);
        break;
      case 'Digit9':
        selectCamera('timesSquare', true);
        break;
      case 'Digit0':
        selectCamera('chryslerGCT', true);
        break;
      case 'KeyV':
        cycleCamera(e.shiftKey ? -1 : 1);
        break;
      case 'KeyH':
        if (hudToggleBtn) hudToggleBtn.click();
        break;
      case 'KeyN':
        if (!e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey) audioSystem.setEnabled(!audioSystem.enabled);
        break;
      case 'KeyB':
        if (!e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey && !mapSystem.isFullMapOpen) audioSystem.honk(controllerManager.getActiveEntity());
        break;
    }
  });

  window.addEventListener('resize', () => {
    if (capturing) return; // the capture sets the window's size again when it ends
    cameraSystem.onResize(window.innerWidth, window.innerHeight);
    renderer.setSize(window.innerWidth, window.innerHeight);
    placeModeIndicator();
    if (mapSystem.isFullMapOpen) {
      mapSystem.resizeFullCanvas();
      mapSystem.renderFullMap();
    }
  });

  // Initialize the UI and chase camera around the restored player, or the default taxi on a first visit.
  selectMode(controllerManager.activeMode);
  selectCamera('chase', true);
  selectTime('golden', true);
  // the web fonts change the buttons' widths once they arrive
  if (document.fonts) document.fonts.ready.then(placeModeIndicator);
  markInit('ui');

  // Compile shaders behind the loading screen. Overlapping compileAsync() jobs keep compilation moving
  // while individual jobs yield to the browser.
  const COMPILE_POOL = 16;
  const nextTask = () => (globalThis.scheduler?.yield ? globalThis.scheduler.yield() : new Promise((resolve) => setTimeout(resolve, 0)));
  async function precompileShaders() {
    await showLoadingStage('Preparing reflections and bloom...', 'postFX');
    // An empty frame first: it sets up the post-processing chain, which configures the scene pass's render target
    // (MSAA samples, depth format), so the pipelines compiled below are the very ones the scene pass asks for
    scene.visible = false;
    renderPipeline.render();
    scene.visible = true;
    markInit('postFX');

    // Every visible object with frustum culling off (like the warm-up frame, see the loop). Plain meshes that share
    // a material share its shader, so one job compiles them in turn (the first build serves the rest); every
    // instanced mesh gets a build of its own unless its chunks share a pipeline
    const culled = [];
    const jobs = [];
    const jobByMaterial = new Map();
    const pipelines = new Set();
    scene.traverseVisible((o) => {
      if (o.frustumCulled) {
        culled.push(o);
        o.frustumCulled = false;
      }
      if (!o.isMesh && !o.isLine && !o.isPoints && !o.isSprite) return;
      // Use the first visible chunk: the original representative may be in an inactive tree cell. Shadow stand-ins
      // only draw in the shadow maps (the warm-up frame compiles those).
      if (o.userData.shadowProxy) return;
      const pipeline = o.userData.sharesPipelineWith || o;
      if (pipelines.has(pipeline)) return;
      pipelines.add(pipeline);
      if (o.isInstancedMesh || Array.isArray(o.material)) {
        jobs.push([o]);
        return;
      }
      let job = jobByMaterial.get(o.material);
      if (!job) {
        job = [];
        jobByMaterial.set(o.material, job);
        jobs.push(job);
      }
      job.push(o);
    });

    await showLoadingStage(`Compiling scene shaders... 0 / ${jobs.length} batches`, 'compile', false);
    // Both passes draw every object, each with shaders of its own: the pre-pass writes its MRT, the beauty pass
    // shades with the SSR and SSGI context
    const compile = async (object) => {
      for (const target of [prePass, beautyPass]) await compileFor(target, object);
    };
    // compileAsync() runs the per-frame updates of the nodes a shader depends on: for the beauty shaders that is the
    // pre-pass, i.e. drawing the whole city once per job (43 s in all). Nothing is displayed yet, so skip them.
    // ponytail: patches a private NodeFrame method; drop once compileAsync() stops running render updates itself
    // The empty frame above drew no material, so nothing asked for the pre-pass yet and its render target still has
    // the default depth format: use the one the pass sets up for a reversed depth buffer, or its pipelines won't match
    prePass.renderTarget.depthTexture.type = THREE.FloatType;
    const nodeFrame = renderer._nodes.nodeFrame;
    nodeFrame.updateBeforeNode = () => {};
    const compileFor = async (target, object) => {
      // Each compile starts in a task of its own: compileAsync() keeps one precompilation state on the renderer,
      // which must not be mid-step for two of them in the same microtask checkpoint
      await nextTask();
      const contextNode = renderer.contextNode;
      renderer.setRenderTarget(target.renderTarget);
      renderer.setMRT(target.getMRT());
      if (target.contextNode !== null) renderer.contextNode = target._getContextNode(renderer);
      try {
        // compileAsync() captures the render target, MRT and context before its first await
        const compiled = renderer.compileAsync(object, cameraSystem.camera, scene);
        renderer.contextNode = contextNode;
        await compiled;
      } catch (e) {
        console.warn('[compile]', e); // the warm-up frame compiles it instead
      }
    };
    let next = 0;
    let done = 0;
    const worker = async () => {
      while (next < jobs.length) {
        const job = jobs[next++];
        for (const object of job) await compile(object);
        done++;
        if (loadingStatus) loadingStatus.textContent = `Compiling scene shaders... ${done} / ${jobs.length} batches`;
        setStageProgress(done / jobs.length);
      }
    };
    await Promise.all(Array.from({ length: COMPILE_POOL }, worker));
    delete nodeFrame.updateBeforeNode;
    renderer.setMRT(null);
    renderer.setRenderTarget(null);
    for (const o of culled) o.frustumCulled = true;
  }

  await precompileShaders();
  markInit('compile');
  await showLoadingStage('Preparing shadows and the first frame...', 'warmup');

  // 6. Animation Loop
  let lastTime = performance.now();
  let elapsed = 0;
  let frameCount = 0;
  let fpsTimer = 0;
  let nextPlayerSave = 0;
  let warmedUp = false; // see the first frame below

  if (elFps) elFps.textContent = '60';

  renderer.setAnimationLoop((now) => {
    const rawDt = (now - lastTime) * 0.001;
    lastTime = now;
    const dt = Math.min(Math.max(rawDt, 0.001), 0.05);
    // A capture holds the world and the camera still, so its frames differ in their lighting alone
    if (capturing) {
      syncSkyOcclusion();
      renderPipeline.render();
      if (afterCaptureFrame) afterCaptureFrame();
      return;
    }
    elapsed += dt;

    // Update 3rd-person controllers, camera, signals, traffic & vehicle collisions, building LODs, and dynamic shadow focus
    gamepadControls.update(dt);
    controllerManager.update(dt, elapsed, cameraSystem.initialized ? cameraSystem.currentPos : null);
    cameraSystem.update(dt, elapsed);
    streetDetails.signalClock.value = elapsed;
    trafficSystem.update(
      dt,
      elapsed,
      cameraSystem.currentPos,
      controllerManager.getActiveEntity(),
      controllerManager.activeMode,
      cameraSystem.camera
    );
    highwayTraffic.update(dt, cameraSystem.camera, controllerManager.getActiveEntity(), controllerManager.activeMode);
    if (!document.hidden && now >= nextPlayerSave) {
      savePlayerState();
      nextPlayerSave = now + 5000;
    }
    if (boroughs.update) boroughs.update(elapsed); // Roosevelt Island Tramway cabins
    cityBlocks.update(cameraSystem.currentPos.x, cameraSystem.currentPos.z);
    if (cityBlocks.updateLights) cityBlocks.updateLights(cameraSystem.camera);
    if (streetDetails.update) streetDetails.update(cameraSystem.camera);
    lightingSystem.update(cameraSystem.currentPos);
    colorGrade.update();
    mapSystem.update();
    audioSystem.update(dt, elapsed, cameraSystem.camera, lightingSystem.activePresetId === 'midnight');
    if (warmedUp) updateSceneCulling(cameraSystem.camera.position); // the warm-up frame draws every chunk
    // Traffic collisions can move the player after controller updates; finalize moving hierarchies here.
    for (const root of dynamicRoots) root.updateMatrixWorld();

    syncSkyOcclusion();
    // Render frame via TSL RenderPipeline (SSR + Bloom)
    if (!warmedUp) {
      // First frame = pipeline warm-up behind the loading screen, for what precompileShaders() can't reach (the shadow
      // map passes): with frustum culling off, every visible material's pipelines compile now instead of stalling the
      // first time the camera turns towards it (one chunk / shadow stand-in per shared pipeline is enough)
      const culled = [];
      const pipelines = new Set();
      scene.traverseVisible((o) => {
        const pipeline = o.userData.sharesPipelineWith || o;
        if (pipelines.has(pipeline)) return;
        pipelines.add(pipeline);
        if (o.frustumCulled) {
          culled.push(o);
          o.frustumCulled = false;
        }
      });
      renderPipeline.render();
      for (const o of culled) o.frustumCulled = true;
      markInit('warmup');
    }
    renderPipeline.render();
    if (afterCaptureFrame) afterCaptureFrame();
    updateHeroProbe();
    if (!warmedUp) {
      warmedUp = true;
      markInit('firstFrame');
      // The loading screen stays up until the GPU has got through the first frame (and the pipelines its warm-up
      // compiled), i.e. until the city is actually on screen
      const device = renderer.backend.device;
      beginLoadingStage('Waiting for graphics to finish the first frame...', 'gpu');
      Promise.resolve(device && device.queue.onSubmittedWorkDone()).catch(() => {}).then(() => {
        markInit('gpu');
        console.log(`[init] ${initTimings.join(', ')}`);
        initTimings = null;
        setLoadingProgress(1);
        if (loadingOverlay) loadingOverlay.classList.add('hidden');
        audioSystem.ready = true;
        window.profilePasses = profilePasses;
      });
    }

    // Telemetry uses wall time, not the clamped simulation step, so slow frames and high-refresh displays report
    // their actual frame rate (updates DOM every 250ms).
    frameCount++;
    fpsTimer += Math.max(0, rawDt);
    if (fpsTimer >= 0.25) {
      const currentFps = Math.round(frameCount / fpsTimer);
      frameCount = 0;
      fpsTimer = 0;

      const activeEnt = controllerManager.getActiveEntity();
      const mode = controllerManager.activeMode;
      const mph = Math.round(Math.abs(activeEnt.speed) * 2.237);
      const altFt = Math.max(0, Math.round(activeEnt.position.y * 3.281));

      if (elFps) {
        elFps.textContent = String(currentFps);
        elFps.className = currentFps >= 50 ? 'fps-ok' : currentFps >= 30 ? 'fps-warn' : 'fps-bad';
      }
      if (elSpeed) elSpeed.textContent = String(mph);
      if (elSpeedMeter) elSpeedMeter.style.transform = `scaleX(${Math.min(1, mph / (topSpeedMph[mode] || 80))})`;
      if (elAltitude) elAltitude.textContent = altFt.toLocaleString('en-US');
      // The street corner of the player on the follow cameras, of the camera itself on the scenic ones
      const where = cameraSystem.getActivePreset().category === 'follow' ? activeEnt.position : cameraSystem.currentPos;
      renderLocation(getLocationLabel(where.x, where.z));
    }
  });
}

initApp();

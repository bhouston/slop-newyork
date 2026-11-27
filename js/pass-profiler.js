import { RendererInspector } from 'three/addons/inspector/RendererInspector.js';

// Where the time of a frame goes, by render and compute call, for any three.js WebGPURenderer driven by
// renderer.setAnimationLoop():
//
//   const profiler = new PassProfiler();
//   renderer.inspector = profiler;
//   console.log(formatPassProfile(await profiler.measure(5)));
//
// three.js times every pass with GPU timestamp queries (the `timestamp-query` feature) and hands each render call
// the GPU timestamps of its first pass's start and its last pass's end. The three.js Inspector shows these spans as
// they are, which don't add up: a call's span holds the calls nested in it (a shadow map in the beauty pass, every
// effect in the output quad), and on a tile-based GPU (Apple, mobile) the passes overlap in time, the vertex stage of
// one starting while the fragment stage of the one before still runs. Here the spans are laid on the GPU's clock:
// each call keeps the part of its span that no call nested in it covers, and a moment several calls cover is split
// between them. The times of the calls then add up to the time the GPU is busy by construction, and the time between
// the starts of successive frames, on the same clock, says how much of a frame that is.
//
// Run it in a browser without a frame-rate limit (tools/profile-passes.mjs): a frame-limited GPU idles between the
// frames and clocks down, which stretches every pass. Chrome rounds the timestamps to 0.1 ms unless it has
// --enable-webgpu-developer-features; the error averages out over the frames but shows in the smallest passes.

// The parts of [a, b] that none of the sorted, disjoint intervals covers
function subtract(a, b, covered) {
  const parts = [];
  for (const [c, d] of covered) {
    if (d <= a || c >= b) continue;
    if (c > a) parts.push([a, c]);
    a = Math.max(a, d);
  }
  if (a < b) parts.push([a, b]);
  return parts;
}

// The sorted, disjoint union of intervals
function union(intervals) {
  const merged = [];
  for (const [a, b] of [...intervals].sort((p, q) => p[0] - q[0])) {
    const last = merged.at(-1);
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else merged.push([a, b]);
  }
  return merged;
}

// A call by the name of what it draws (a scene, a QuadMesh) and the size of what it draws to
function passLabel(stats) {
  if (!stats.isRenderStats) return stats.name;
  const target = stats.renderTarget;
  return target ? `${stats.name} ${target.width}x${target.height}` : `${stats.name} (canvas)`;
}

// The Inspector files the consecutive calls named "Group / Pass" under a group of that name. Here they are the
// calls of their parent again, each under its full name: [call, the names of its groups]
function ungroup(children, prefix = '') {
  return children.flatMap((child) => child.isGroupStats ? ungroup(child.children, `${prefix}${child.name} / `) : [[child, prefix]]);
}

class PassProfiler extends RendererInspector {
  // label(stats): the name of a call, from its stats (name, scene, camera, renderTarget, computeNode, parent)
  constructor({ label = passLabel } = {}) {
    super();
    this.label = label;
    this._run = null;
  }

  begin() {
    super.begin();
    const run = this._run;
    this.currentFrame.run = run;
    if (run) run.pending++;
  }

  // The draw calls and triangles of a render call, with those of the calls nested in it
  beginRender(uid, scene, camera, renderTarget) {
    super.beginRender(uid, scene, camera, renderTarget);
    const { drawCalls, triangles } = this.getRenderer().info.render;
    if (this.currentRender) Object.assign(this.currentRender, { draws: -drawCalls, triangles: -triangles });
  }

  finishRender(uid) {
    const stats = this.currentRender;
    const { drawCalls, triangles } = this.getRenderer().info.render;
    if (stats) {
      stats.draws += drawCalls;
      stats.triangles += triangles;
    }
    super.finishRender(uid);
  }

  // As RendererInspector's, keeping the timestamps and not only their difference
  _endTimer(frame, stats) {
    frame.pendingTimers++;
    stats.timer.end().then((gpu) => {
      if (gpu === null) stats.gpuNotAvailable = true;
      else Object.assign(stats, { gpu: gpu.delta, gpuStart: gpu.start, gpuEnd: gpu.end });
      frame.pendingTimers--;
      this._resolveFrameIfComplete(frame);
    });
  }

  // Called once per frame, when the GPU has answered all of its timers
  resolveFrame(frame) {
    const run = frame.run;
    if (!run) return;
    frame.run = null;
    if (frame.children.length > 0) this._add(run, frame);
    if (--run.pending === 0 && run.ended) run.resolve();
  }

  _add(run, frame) {
    const segments = []; // [start, end, row]: the part of a call's span that is its own
    const visit = (stats, path) => {
      const names = new Map();
      for (const [child, groups] of ungroup(stats.children)) {
        const nestedCalls = ungroup(child.children).map(([call]) => call);
        let name = groups + this.label(child);
        const count = (names.get(name) || 0) + 1;
        names.set(name, count);
        if (count > 1) name += ` #${count}`; // the calls of a frame that share a name, in order
        const key = path ? `${path} > ${name}` : name;
        let row = run.rows.get(key);
        if (!row) run.rows.set(key, row = { name: key, depth: key.split(' > ').length - 1, calls: 0, draws: 0, triangles: 0, cpu: 0, gpu: 0, span: 0 });
        row.calls++;
        const own = (key) => (child[key] || 0) - nestedCalls.reduce((sum, c) => sum + (c[key] || 0), 0);
        row.cpu += own('cpu');
        row.draws += own('draws');
        row.triangles += own('triangles');
        if (child.gpuNotAvailable) run.missing++;
        if (child.gpuStart != null) {
          if ((child.gpuStart * 10) % 1 > 1e-3) run.quantized = false;
          const nested = union(nestedCalls.filter((c) => c.gpuStart != null).map((c) => [c.gpuStart, c.gpuEnd]));
          for (const [a, b] of subtract(child.gpuStart, child.gpuEnd, nested)) {
            segments.push([a, b, row]);
            row.span += b - a;
            run.span += b - a;
          }
        }
        visit(child, key);
      }
    };
    visit(frame, '');
    run.frames++;
    run.cpuFrame += frame.finishTime - frame.startTime;
    if (segments.length === 0) return;

    // Sweep the GPU's clock: a moment that k calls cover counts 1 / k for each of them
    const edges = [...new Set(segments.flatMap(([a, b]) => [a, b]))].sort((a, b) => a - b);
    for (let i = 0; i + 1 < edges.length; i++) {
      const a = edges[i], b = edges[i + 1];
      const active = segments.filter((s) => s[0] <= a && s[1] >= b);
      for (const s of active) s[2].gpu += (b - a) / active.length;
      if (active.length > 0) run.busy += b - a;
    }
    const start = edges[0];
    if (run.firstStart === null) run.firstStart = start;
    run.lastStart = start;
    run.timed++;
  }

  // Collects the frames of the next `seconds` and resolves with their averages, in milliseconds per frame. Warm
  // the scene up first.
  async measure(seconds = 5) {
    const queue = this.getRenderer().backend.device?.queue; // WebGPU only
    if (queue) await queue.onSubmittedWorkDone(); // the work queued before the measurement is not part of it
    const run = {
      rows: new Map(), frames: 0, timed: 0, pending: 0, ended: false, resolve: null, missing: 0, quantized: true,
      cpuFrame: 0, busy: 0, span: 0, firstStart: null, lastStart: null
    };
    const settled = new Promise((resolve) => run.resolve = resolve);
    const t0 = performance.now();
    this._run = run;
    await new Promise((resolve) => setTimeout(resolve, seconds * 1000));
    this._run = null;
    const begun = run.frames + run.pending;
    if (queue) await queue.onSubmittedWorkDone(); // count the frames the GPU has completed, not the ones submitted
    const elapsed = performance.now() - t0;
    run.ended = true;
    if (run.pending > 0) await Promise.race([settled, new Promise((resolve) => setTimeout(resolve, 3000))]);

    const n = Math.max(run.frames, 1), timed = Math.max(run.timed, 1);
    const period = run.timed > 1 ? (run.lastStart - run.firstStart) / (run.timed - 1) : null;
    const info = (await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }))?.info;
    const size = this.getRenderer().domElement;
    return {
      adapter: info ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' ') : 'unknown',
      size: `${size.width} x ${size.height}`,
      seconds: elapsed / 1000,
      frames: begun,
      fps: begun / (elapsed / 1000),
      frameMs: elapsed / begun, // wall clock, per GPU-completed frame
      cpuFrameMs: run.cpuFrame / n, // the animation loop's callback: the app's update and every render call
      gpu: run.timed === 0 ? null : {
        periodMs: period, // between the starts of successive frames, on the GPU's clock
        busyMs: run.busy / timed, // some pass is running
        idleMs: period === null ? null : period - run.busy / timed,
        overlapMs: (run.span - run.busy) / timed, // the passes that run at the same time, counted once in busyMs
        quantized: run.quantized, // the timestamps are rounded to 0.1 ms
        untimedCalls: run.missing
      },
      // in the order of the frame, per frame: the calls, the CPU time issuing them, the GPU time (which adds up to
      // busyMs) and the GPU span as measured (which adds up to busyMs + overlapMs)
      passes: [...run.rows.values()].map((row) => ({
        name: row.name, depth: row.depth, calls: row.calls / n, draws: row.draws / n, triangles: row.triangles / n, cpuMs: row.cpu / n, gpuMs: row.gpu / timed, gpuSpanMs: row.span / timed
      }))
    };
  }
}

function formatPassProfile(report) {
  const ms = (value) => value == null ? '-' : value.toFixed(2).padStart(7);
  const lines = [
    `${report.adapter}, ${report.size}`,
    `${report.frames} frames in ${report.seconds.toFixed(2)} s: ${report.fps.toFixed(1)} fps, ${report.frameMs.toFixed(2)} ms per frame`,
    ''
  ];
  const { gpu } = report;
  if (gpu === null) return [...lines, 'No GPU timestamps: the timestamp-query feature is not available.'].join('\n');
  const rows = report.passes.map((p) => {
    const name = p.name.split(' > ').pop();
    return [`${'  '.repeat(p.depth)}${name}`, p.calls.toFixed(p.calls % 1 ? 2 : 0), Math.round(p.draws).toString().padStart(6), ms(p.cpuMs), ms(p.gpuMs), ms(p.gpuSpanMs), `${(p.gpuMs / gpu.busyMs * 100).toFixed(1).padStart(5)}%`];
  });
  const width = Math.max(28, ...rows.map((r) => r[0].length));
  const line = (r) => `${r[0].padEnd(width)}  ${r[1].padStart(5)}  ${r[2]}  ${r[3]}  ${r[4]}  ${r[5]}  ${r[6]}`;
  lines.push(line(['call', 'calls', ' draws', ' cpu ms', ' gpu ms', 'span ms', 'of gpu']), ...rows.map(line), '');
  const draws = report.passes.reduce((sum, p) => sum + p.draws, 0), triangles = report.passes.reduce((sum, p) => sum + p.triangles, 0);
  const cpuCalls = report.passes.reduce((sum, p) => sum + p.cpuMs, 0);
  lines.push(
    `GPU busy       ${ms(gpu.busyMs)} ms   the sum of the gpu ms column`,
    `GPU idle       ${ms(gpu.idleMs)} ms   between the passes and between the frames`,
    `GPU period     ${ms(gpu.periodMs)} ms   start of a frame to the start of the next, on the GPU's clock`,
    `Frame (wall)   ${ms(report.frameMs)} ms`,
    `GPU overlap    ${ms(gpu.overlapMs)} ms   passes running at the same time: span ms counts it for each of them`,
    `CPU frame      ${ms(report.cpuFrameMs)} ms   the animation loop's callback, of which ${cpuCalls.toFixed(2)} ms issue the calls above`,
    `Draw calls     ${Math.round(draws).toString().padStart(7)}      ${(triangles / 1e6).toFixed(2)} million triangles`
  );
  if (gpu.quantized) lines.push('The timestamps are rounded to 0.1 ms: start the browser with --enable-webgpu-developer-features.');
  if (gpu.untimedCalls > 0) lines.push(`${gpu.untimedCalls} calls have no GPU time (their timers were lost).`);
  return lines.join('\n');
}

export { PassProfiler, passLabel, formatPassProfile };

# Where a frame goes: profiling report

November 2026. Measured on the `ssr-brdf` branch with the uncommitted profiler described below.

## Summary

- The frame is limited mostly by the CPU (the JavaScript main thread): issuing the render calls sets a floor of about 27 ms per frame at any resolution. At 1920×993 the GPU adds about 6 ms on top, as stalls: roughly one frame in seven takes twice as long.
- About 31 ms of that is spent issuing render calls, and almost all of it in seven scene renders: the pre-pass, the beauty pass, one face of the hero probe, and four shadow-map renders.
- Every post effect combined (SSR, SSGI, denoise, TRAA, FSR1, bloom) costs about 1.8 ms of CPU. Their GPU cost is the smaller part of the frame: rendering 9× fewer pixels raises the frame rate from 30 to 35 fps.
- The three.js Inspector's per-pass GPU times do not add up for two reasons: it shows nested calls inside their parents, and on Apple GPUs the passes overlap in time. Removing the browser's frame-rate limit does not fix either.
- On this GPU, per-pass GPU timestamps cannot give a trustworthy per-pass cost at all. CPU times and draw counts are exact and repeatable.

## Findings

Test conditions: headless Chrome 158 with vsync and the frame-rate limit disabled, Apple M3 (Metal), 1920×993 canvas, the default spawn point and time of day, 5-second measurement after a 6-second warm-up. Without the profiler attached the scene ran at 29.7 fps (33.6 ms per frame).

### CPU time per frame

| Render call | Draw calls | CPU ms |
|---|---|---|
| Pre-pass | 778 | 9.4 |
| Beauty pass | 778 | 8.7 |
| Hero probe (one cube face per frame) | 544 | 6.7 |
| Shadow maps for the main view (2 cascades) | 372 | 2.4 |
| Shadow maps for the hero probe's camera (2 cascades) | 363 | 2.4 |
| All post effects (about 30 full-screen passes) | about 30 | 1.8 |
| **Render calls total** | **2,863** | **31.4** |
| App update (traffic, pedestrians, culling, audio, UI) | – | 2.0 |
| **Animation-loop callback** | | **33.4** |

The frame draws 11.8 million triangles. A scene render costs roughly 12 µs of CPU per draw call.

### How much is CPU and how much is GPU

- At 640×273, 9× fewer pixels, the unprofiled frame rate only rose from about 30 to 35.5 fps (33 to 28 ms). About 27–28 ms of the frame does not depend on resolution.
- The frame intervals at 640×273 are all between 25 and 37 ms. At 1920×993 most frames are in the same range, but 27 of 200 took 54–67 ms. Those stalls are what raises the average from about 28 to 33 ms, and they only appear at the larger size, so they are the GPU falling behind.
- The callback's wall-clock time includes those stalls: the main thread blocks inside a render call when the GPU is behind. The "CPU ms" column above is therefore slightly inflated at 1920×993, mostly in the pre-pass, the first render of the frame. At 640×273 the callback takes 27.0 ms.
- Removing 350 draw calls and about 2.4 ms of render-call time (see the shadow-map change below) changed the frame rate at 1920×993 by less than the run-to-run noise. That is consistent with the GPU, not the CPU, setting the last few milliseconds at that size.
- Earlier in this work, switching off the sun's cascaded shadow maps "sped up the scene tremendously". That removed four scene traversals.

I measured the callback with wall-clock time, not a sampling CPU profile, so I cannot yet separate real JavaScript work from time blocked on the GPU process within a call (see "Next steps").

### Two things worth knowing about the frame

- **The shadow cascades rendered twice per frame.** three.js fits the cascades to the camera of each render and redraws them when a different camera renders, so the hero probe's render triggered its own pair: 363 draws and 2.4 ms of render-call time per frame. This is now fixed: the probe reuses the frame's cascades. Its shadows are right around the vehicle and in front of the camera, and not reliable behind the camera, where the view's cascades do not reach.
- **The hero probe cost about 9 ms per frame** (its render plus its shadow maps), 27% of the frame, for the reflections on one vehicle. Without its own shadow maps it is about 6 ms.

### GPU timestamps on this machine

| | ms per frame |
|---|---|
| GPU period (start of one frame to the start of the next, GPU clock) | 33.6 |
| Time some pass is running | 33.5 |
| Sum of the raw pass spans, beyond that | 113.3 |

The raw spans sum to more than four times the frame. Individual values are not believable as costs:

- A bloom blur at 42×21 pixels reports a 3.4 ms span.
- Each SSGI denoise pass at 720×372 reports about 11 ms.
- The pre-pass reports 5.4 ms at 1920×993 and 11.7 ms at 640×273.

A tile-based GPU runs the vertex stage of later passes while the fragment stage of earlier ones is still in flight, so a pass's start-to-end interval covers its neighbours. When the frame is CPU-bound, the intervals also stretch to cover the time the GPU waits for the next submission.

## How the profiling was done

### What three.js provides

`WebGPURenderer` requests the WebGPU `timestamp-query` feature and writes a begin and end timestamp for every render and compute pass. `GPUTimer` returns the earliest start and latest end of the passes issued between `begin()` and `end()`. The renderer calls `beginRender` / `finishRender` hooks on `renderer.inspector` around every render call, nested calls included, and `RendererInspector` uses these to build a per-frame tree with a CPU time and a GPU span for each call.

### What the profiler adds

[js/pass-profiler.js](js/pass-profiler.js) is a headless subclass of `RendererInspector`. It has no UI and works with any three.js WebGPU app driven by `renderer.setAnimationLoop()`.

- **Exclusive CPU time:** a call's CPU time minus that of the calls nested in it.
- **Draw calls and triangles per call**, from `renderer.info`.
- **GPU intervals laid on the GPU's clock:** each call keeps the part of its span that no nested call covers. A moment covered by several calls is split equally between them, so the GPU column sums to the busy time by construction.
- **Check lines:** GPU busy, idle, period, and overlap (how much the raw spans exceed the busy time). Overlap near zero means the GPU column is trustworthy; a large overlap means it is an estimate.
- **Averages over a time window** (default 5 seconds), counting frames the GPU has completed, not single-frame readings.

[tools/profile-passes.mjs](tools/profile-passes.mjs) makes the run repeatable. It launches Chrome with a temporary profile, drives it over the DevTools protocol, and needs no dependencies beyond Node. It:

1. Starts Chrome with `--disable-gpu-vsync --disable-frame-rate-limit`, background throttling off, a fixed window size and scale factor, and `--enable-webgpu-developer-features` (without it Chrome rounds timestamps to 0.1 ms).
2. Waits for the page to define `window.profilePasses`, then waits for the warm-up.
3. Measures the sustained frame rate without the profiler.
4. Calls `profilePasses(seconds)` and prints the table.
5. Refuses the result if the adapter is a software rasterizer.

```
node tools/profile-passes.mjs --headless
node tools/profile-passes.mjs --headless --size 640x360     # is it resolution-bound?
node tools/profile-passes.mjs --vsync                       # keep the frame limit, for comparison
```

In this app, `profilePasses(5)` also works from the browser console once the city is on screen.

### Limits of these measurements

- One machine, one GPU architecture. I could not check that the GPU column is accurate on a desktop GPU that runs passes one after another.
- One camera position and time of day.
- Headless Chrome only. A run in a visible window never finished loading under the script, and I did not find out why. Headless is not pacing frames to the display: the frame intervals are not multiples of 16.7 ms.
- A second Chrome running at the same time slows the measured one by a third or more; run nothing else on the GPU.
- Attaching the profiler costs about 10% (37.5 against 33.6 ms per frame).
- Three full-size runs gave 25.7, 26.7 and 26.8 fps with the profiler attached, so repeatability is within about 4%.

## How three.js's built-in profiling could be improved

1. **Show exclusive time.** The Performance tab lists each call's inclusive GPU span, so a parent and its children are both counted. Subtracting the nested calls, or labelling the column as inclusive, would remove the largest source of confusion.
2. **Keep the timestamps.** `GPUTimer.end()` already returns `start` and `end`, but `RendererInspector._endTimer` keeps only `delta`. With the absolute timestamps, the Inspector can detect overlap and compute the frame's busy and idle time.
3. **Detect and report overlap.** If the sum of the leaf spans exceeds their union, the GPU is running passes concurrently and per-pass times are not additive. The tab should say so instead of presenting the numbers as costs.
4. **Stop clamping.** The tab clamps each GPU value to its parent's budget (`Math.min( data.gpu, budget )`), which hides the inconsistency.
5. **Say whether the frame is CPU- or GPU-bound.** Comparing the callback's duration with the frame period answers the first question anyone has, and it needs no GPU timestamps.
6. **Add draw calls and triangles per call.** These explain CPU cost directly.
7. **Average over a window and allow export.** A last-frame readout is noisy; a `measure(seconds)` API returning JSON would make results comparable between runs and machines.
8. **Warn about conditions that distort the numbers:** timestamps rounded to 0.1 ms, and a frame-rate-limited browser.
9. **Document the tile-based GPU limitation** on `GPUTimer` and the Performance tab.

Items 1, 2, 5 and 6 are small changes. [js/pass-profiler.js](js/pass-profiler.js) implements all of 1–3 and 5–7 in about 200 lines and could be the basis of an upstream change.

## Recommendations

### For this project

The render calls are the larger part of the frame, so start there; the GPU's share only shows at full resolution. In order of expected gain:

1. **Hero probe (about 6 ms now that it reuses the frame's shadow maps).** Update a face every few frames instead of every frame, or draw a reduced set of objects into it.
2. **Pre-pass and beauty pass (about 18 ms).** Both traverse and encode the same 778 draws. Reducing the draw count (merging static chunks further, batching) helps both at once.
3. **Shadow maps (about 4.8 ms).** Update the far cascade every other frame, and cull casters harder.
4. **Find out why a draw costs 12 µs.** A JavaScript CPU profile of one frame would show how much is traversal, culling, sorting, per-object uniform updates or command encoding, and whether the cost is in the app or in three.js.

### Next steps for the tooling

- **Frame-interval histogram in the report.** The average hides the stalls; the distribution is what showed the GPU's share.
- **CPU sampling profile from the same script.** The DevTools protocol can record one; this would break the 9 ms per scene render down by function and remove the caveat above.
- **Ablation mode for GPU cost.** Switch each effect off in turn and compare sustained frame rate. This is the only method that gives trustworthy per-effect GPU costs on a tile-based GPU. It needs each effect to have a real off switch, and it is only meaningful once the frame is GPU-bound.
- **Run on a desktop discrete GPU** to confirm that overlap is near zero there and the GPU column can be trusted.
- **A fixed camera path** so that runs on different days compare like with like.

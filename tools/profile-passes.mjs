// Where the time of a frame goes, by pass: opens a page in Chrome without a frame-rate limit, lets it warm up,
// calls its profilePasses(seconds) (see js/pass-profiler.js) and prints the report.
//
//   node tools/profile-passes.mjs [url] [--seconds 5] [--warmup 8] [--size 1920x1080] [--vsync] [--headless] [--json]
//
// Any page that defines window.profilePasses(seconds), resolving with { text, ... }, can be profiled. The page is
// ready when that function exists. --vsync keeps the display's frame-rate limit, for comparison: the GPU then idles
// between the frames and clocks down. CHROME_PATH overrides the browser.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const option = (name, fallback) => args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : fallback;
const url = args.find((a, i) => !a.startsWith('--') && !(args[i - 1] || '').match(/^--(seconds|warmup|size)$/)) || 'http://localhost:8000/';
const seconds = Number(option('seconds', 5));
const warmup = Number(option('warmup', 8));
const [width, height] = option('size', '1920x1080').split('x').map(Number);
const chrome = process.env.CHROME_PATH || {
  darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  win32: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
}[process.platform] || 'google-chrome';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const profileDir = mkdtempSync(join(tmpdir(), 'profile-passes-'));
const port = 9300 + Math.floor(Math.random() * 600);
const browser = spawn(chrome, [
  `--remote-debugging-port=${port}`, `--user-data-dir=${profileDir}`, '--no-first-run', '--no-default-browser-check',
  `--window-size=${width},${height}`, '--force-device-scale-factor=1',
  // keep rendering at full rate when the window is covered or in the background
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  '--enable-webgpu-developer-features', // timestamps that are not rounded to 0.1 ms
  ...(flag('vsync') ? [] : ['--disable-gpu-vsync', '--disable-frame-rate-limit']),
  ...(flag('headless') ? ['--headless=new'] : []),
  'about:blank'
], { stdio: 'ignore' });

let failure = null;
try {
  let target;
  for (let i = 0; i < 100 && !target; i++) {
    await sleep(100);
    target = await fetch(`http://127.0.0.1:${port}/json/list`).then((r) => r.json()).then((list) => list.find((t) => t.type === 'page')).catch(() => null);
  }
  if (!target) throw new Error(`Chrome did not start: ${chrome}`);

  // The Chrome DevTools Protocol over its WebSocket
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  const pending = new Map();
  let id = 0;
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.method === 'Runtime.exceptionThrown') console.error(`[page] ${message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text}`);
    if (pending.has(message.id)) pending.get(message.id)(message);
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++id, (message) => message.error ? reject(new Error(message.error.message)) : resolve(message.result));
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const { result, exceptionDetails } = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || exceptionDetails.text);
    return result.value;
  };

  await send('Runtime.enable');
  await send('Page.navigate', { url });
  const deadline = Date.now() + 180000;
  while (!(await evaluate(`typeof window.profilePasses === 'function'`).catch(() => false))) {
    if (Date.now() > deadline) throw new Error('The page did not define window.profilePasses within 3 minutes.');
    await sleep(500);
  }
  await sleep(warmup * 1000);
  // The sustained frame rate without the profiler, to see what its timers cost: frames over several seconds
  const plain = await evaluate(`new Promise((resolve) => {
    let frames = 0;
    const t0 = performance.now();
    const tick = () => {
      frames++;
      if (performance.now() - t0 < ${seconds * 1000}) requestAnimationFrame(tick);
      else resolve({ frames, ms: performance.now() - t0 });
    };
    requestAnimationFrame(tick);
  })`);
  const report = await evaluate(`window.profilePasses(${seconds})`);
  report.unprofiled = { fps: plain.frames / plain.ms * 1000, frameMs: plain.ms / plain.frames };
  console.log(flag('json') ? JSON.stringify(report, null, 2) : [
    `${flag('vsync') ? 'Frame-rate limited' : 'No frame-rate limit'}, ${url}`,
    report.text,
    `Unprofiled     ${report.unprofiled.frameMs.toFixed(2).padStart(7)} ms   ${report.unprofiled.fps.toFixed(1)} fps before the profiler was attached`
  ].join('\n'));
  if (/swiftshader|llvmpipe|lavapipe/i.test(report.adapter)) failure = 'This is a software rasterizer: the numbers mean nothing.';
} catch (error) {
  failure = error.message;
} finally {
  browser.kill();
  await sleep(300);
  rmSync(profileDir, { recursive: true, force: true, maxRetries: 5 });
}
if (failure) {
  console.error(failure);
  process.exit(1);
}

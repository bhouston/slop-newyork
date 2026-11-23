const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const { loadModule } = require('./helpers/module.cjs');

const { animatedWebP } = loadModule(readFileSync(join(__dirname, '../js/capture.js'), 'utf8'), {
  globals: { Uint8Array, DataView, String, Error }
});

const ascii = (s) => [...s].map((c) => c.charCodeAt(0));
const u32 = (v) => [v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >> 24) & 255];
// A still as the browser encodes it: RIFF, WEBP and one bitstream chunk (padded when its size is odd)
const still = (payload) => {
  const chunk = [...ascii('VP8 '), ...u32(payload.length), ...payload, ...(payload.length & 1 ? [0] : [])];
  return Uint8Array.from([...ascii('RIFF'), ...u32(4 + chunk.length), ...ascii('WEBP'), ...chunk]);
};

test('animatedWebP wraps each still in a frame of an animated container', () => {
  const file = animatedWebP([still([1, 2, 3]), still([4, 5, 6, 7])], 1920, 1080, 1500);
  const view = new DataView(file.buffer);
  const tag = (at) => String.fromCharCode(...file.subarray(at, at + 4));
  const u24 = (at) => file[at] | (file[at + 1] << 8) | (file[at + 2] << 16);

  assert.equal(tag(0), 'RIFF');
  assert.equal(view.getUint32(4, true), file.length - 8);
  assert.equal(tag(8), 'WEBP');

  const chunks = [];
  for (let at = 12; at < file.length;) {
    const size = view.getUint32(at + 4, true);
    chunks.push({ name: tag(at), at: at + 8, size });
    at += 8 + size + (size & 1);
  }
  assert.deepEqual(chunks.map((c) => c.name), ['VP8X', 'ANIM', 'ANMF', 'ANMF']);

  const [header, anim, first, second] = chunks;
  assert.equal(file[header.at], 0x02); // animated, no alpha
  assert.deepEqual([u24(header.at + 4) + 1, u24(header.at + 7) + 1], [1920, 1080]);
  assert.equal(view.getUint16(anim.at + 4, true), 0); // loops forever

  assert.deepEqual([u24(first.at + 6) + 1, u24(first.at + 9) + 1, u24(first.at + 12)], [1920, 1080, 1500]);
  assert.equal(tag(first.at + 16), 'VP8 ');
  assert.deepEqual([...file.subarray(first.at + 24, first.at + 27)], [1, 2, 3]);
  assert.equal(first.size, 16 + 8 + 4); // the odd payload keeps its padding inside the frame
  assert.deepEqual([...file.subarray(second.at + 24, second.at + 28)], [4, 5, 6, 7]);
});

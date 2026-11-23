// Joins still WebP images of one size (as canvas.toBlob('image/webp') encodes them) into an animated WebP that
// loops forever: the container only wraps each still's bitstream in a frame chunk, so nothing is re-encoded.
// https://developers.google.com/speed/webp/docs/riff_container#animation
function animatedWebP(stills, width, height, frameMs) {
  const u24 = (v) => [v & 255, (v >> 8) & 255, (v >> 16) & 255];
  const chunk = (fourcc, payload) => {
    const out = new Uint8Array(8 + payload.length + (payload.length & 1)); // padded to an even size
    out.set([...fourcc].map((c) => c.charCodeAt(0)));
    new DataView(out.buffer).setUint32(4, payload.length, true);
    out.set(payload, 8);
    return out;
  };
  const concat = (parts) => {
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) { out.set(p, at); at += p.length; }
    return out;
  };

  let alpha = false;
  const frames = stills.map((still) => {
    const tag = (at) => String.fromCharCode(...still.subarray(at, at + 4));
    if (tag(0) !== 'RIFF' || tag(8) !== 'WEBP') throw new Error('Not a WebP image');
    const view = new DataView(still.buffer, still.byteOffset, still.byteLength);
    const image = []; // the still's bitstream chunks, without its own header (VP8X) or metadata
    for (let at = 12; at + 8 <= still.length;) {
      const size = view.getUint32(at + 4, true);
      const end = at + 8 + size + (size & 1);
      if (['ALPH', 'VP8 ', 'VP8L'].includes(tag(at))) image.push(still.subarray(at, end));
      if (tag(at) === 'ALPH') alpha = true;
      at = end;
    }
    // at (0, 0), the size of the canvas, shown for frameMs, replacing the frame before it (no blending, no disposal)
    return chunk('ANMF', concat([Uint8Array.from([0, 0, 0, 0, 0, 0, ...u24(width - 1), ...u24(height - 1), ...u24(frameMs), 2]), ...image]));
  });

  const body = concat([
    chunk('VP8X', Uint8Array.from([alpha ? 0x12 : 0x02, 0, 0, 0, ...u24(width - 1), ...u24(height - 1)])), // animated
    chunk('ANIM', new Uint8Array(6)), // background colour 0, loop count 0: forever
    ...frames
  ]);
  const file = concat([chunk('RIFF', concat([Uint8Array.from([0x57, 0x45, 0x42, 0x50]), body]))]); // 'WEBP'
  return file;
}

export { animatedWebP };

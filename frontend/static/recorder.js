/* recorder.js
 * Minimal in-browser recorder: captures raw PCM via Web Audio API and
 * encodes it to a 16-bit PCM WAV Blob directly, client-side. This avoids
 * needing ffmpeg/pydub on the server to decode webm/ogg from MediaRecorder --
 * the backend only ever has to parse plain WAV.
 */
class WavRecorder {
  constructor() {
    this.audioContext = null;
    this.processor = null;
    this.input = null;
    this.stream = null;
    this.buffers = [];
    this.sampleRate = 44100;
    this.recording = false;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.sampleRate = this.audioContext.sampleRate;
    this.input = this.audioContext.createMediaStreamSource(this.stream);

    const bufferSize = 4096;
    this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    this.buffers = [];

    this.processor.onaudioprocess = (e) => {
      if (!this.recording) return;
      const channelData = e.inputBuffer.getChannelData(0);
      this.buffers.push(new Float32Array(channelData));
    };

    this.input.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
    this.recording = true;
  }

  stop() {
    this.recording = false;
    if (this.processor) { this.processor.disconnect(); }
    if (this.input) { this.input.disconnect(); }
    if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); }

    const totalLength = this.buffers.reduce((sum, b) => sum + b.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const b of this.buffers) { merged.set(b, offset); offset += b.length; }

    const wavBlob = encodeWav(merged, this.sampleRate);
    const durationSec = totalLength / this.sampleRate;

    if (this.audioContext) this.audioContext.close();
    return { blob: wavBlob, durationSec };
  }
}

function encodeWav(samples, sampleRate) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

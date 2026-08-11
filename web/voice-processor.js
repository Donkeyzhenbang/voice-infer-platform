/**
 * AudioWorklet 处理器 — 麦克风采集 + 播放双模式。
 *
 * 麦克风模式：128-sample 低延迟采集（~8ms vs ScriptProcessor 256ms）
 * 播放模式：RingBuffer 无间隙播放
 */

const SR = 16000;
const RING_SIZE = SR * 10; // 10s ring buffer

class VoiceProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    this.mode = options.processorOptions?.mode || "mic"; // "mic" | "play"

    if (this.mode === "play") {
      // 环形缓冲区：int16 samples
      this._ring = new Int16Array(RING_SIZE);
      this._readPos = 0;
      this._writePos = 0;
      this._avail = 0;
    }

    // 接收主线程消息
    this.port.onmessage = (e) => {
      if (this.mode === "play" && e.data instanceof ArrayBuffer) {
        // 写入 ring buffer
        const samples = new Int16Array(e.data);
        this._write(samples);
      } else if (e.data === "reset") {
        if (this.mode === "play") {
          this._readPos = 0;
          this._writePos = 0;
          this._avail = 0;
        }
      }
    };
  }

  _write(samples) {
    const free = RING_SIZE - this._avail;
    if (samples.length > free) {
      // 缓冲区满，丢弃最旧数据
      const skip = samples.length - free;
      this._readPos = (this._readPos + skip) % RING_SIZE;
      this._avail -= skip;
    }
    for (let i = 0; i < samples.length; i++) {
      this._ring[this._writePos] = samples[i];
      this._writePos = (this._writePos + 1) % RING_SIZE;
    }
    this._avail = Math.min(RING_SIZE, this._avail + samples.length);
  }

  process(inputs, outputs) {
    if (this.mode === "mic") {
      // 麦克风模式：采集并发送到主线程
      const input = inputs[0];
      if (input && input.length > 0) {
        const channel = input[0];
        if (channel) {
          const i16 = new Int16Array(channel.length);
          for (let i = 0; i < channel.length; i++) {
            i16[i] = Math.max(-32768, Math.min(32767, Math.round(channel[i] * 32767)));
          }
          this.port.postMessage(i16.buffer, [i16.buffer]);
        }
      }
      return true;
    }

    // 播放模式：从 ring buffer 读取并输出
    const output = outputs[0];
    if (output && output.length > 0) {
      const channel = output[0];
      const n = channel.length;
      if (this._avail >= n) {
        // 连续读取
        for (let i = 0; i < n; i++) {
          channel[i] = this._ring[this._readPos] / 32768.0;
          this._readPos = (this._readPos + 1) % RING_SIZE;
        }
        this._avail -= n;
      } else if (this._avail > 0) {
        // 部分可用 → 播放剩余 + 填充静音
        for (let i = 0; i < this._avail; i++) {
          channel[i] = this._ring[this._readPos] / 32768.0;
          this._readPos = (this._readPos + 1) % RING_SIZE;
        }
        for (let i = this._avail; i < n; i++) {
          channel[i] = 0;
        }
        this._avail = 0;
      }
      // 无数据时输出静音（不填零，channel 默认就是 0）
    }
    return true;
  }
}

registerProcessor("voice-processor", VoiceProcessor);

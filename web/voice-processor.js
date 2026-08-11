/**
 * AudioWorklet 处理器 — 麦克风采集 + 播放双模式。
 *
 * 麦克风模式：128-sample 低延迟采集（~8ms vs ScriptProcessor 256ms）
 * 播放模式：RingBuffer 无间隙播放
 */

const SR = 16000;

class VoiceProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    this.mode = options.processorOptions?.mode || "mic"; // "mic" | "play"

    if (this.mode === "play") {
      this._buf = new Int16Array(0);
      this._readPos = 0;
    }

    // 接收主线程消息
    this.port.onmessage = (e) => {
      if (this.mode === "play" && e.data instanceof ArrayBuffer) {
        const samples = new Int16Array(e.data);
        const merged = new Int16Array(this._buf.length + samples.length);
        merged.set(this._buf); merged.set(samples, this._buf.length);
        this._buf = merged;
      } else if (e.data === "reset") {
        if (this.mode === "play") { this._buf = new Int16Array(0); this._readPos = 0; }
      }
    };
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

    // 播放模式：从动态缓冲区读取输出
    const output = outputs[0];
    if (output && output.length > 0) {
      const channel = output[0];
      const n = channel.length;
      const avail = this._buf.length - this._readPos;
      if (avail >= n) {
        for (let i = 0; i < n; i++) {
          channel[i] = this._buf[this._readPos + i] / 32768.0;
        }
        this._readPos += n;
      } else if (avail > 0) {
        for (let i = 0; i < avail; i++) {
          channel[i] = this._buf[this._readPos + i] / 32768.0;
        }
        this._readPos += avail;
      }
    }
    return true;
  }
}

registerProcessor("voice-processor", VoiceProcessor);

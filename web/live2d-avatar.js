/**
 * Live2D 卡通猫头像 — 音频包络驱动口型（ParamMouthOpenY）
 *
 * 基于 pixi-live2d-display + CDN 猫模型（hijiki/nico）
 * 口型驱动参考 xiaozhi-esp32-server 的音量包络方案
 *
 * 接口与 Canvas Avatar 兼容：
 *   const avatar = new Live2DAvatar(canvas, fallbackCanvas);
 *   avatar.feedAudio(int16Bytes);
 *   avatar.setState('idle'|'speaking'|'thinking'|'listening');
 */
class Live2DAvatar {
  constructor(fallbackCanvas) {
    this.fallback = null;
    if (fallbackCanvas && window.Avatar) {
      this.fallback = new window.Avatar(fallbackCanvas);
    }
    this.model = null;
    this.state = "idle";
    this.smoothRMS = 0;
    this.mouthOpenY = 0;
    this.isTalking = false;
    this.ready = false;
    this.raf = null;
    // 模型配置（不同模型音量响应差异大）
    this.cfg = { low: 0.08, high: 0.25, minOpenY: 0.05, maxOpenY: 1.0 };
  }

  /**
   * 加载 Live2D 模型（CDN），失败则回退 Canvas。
   * @param {string} modelUrl - model.json 地址
   */
  async load(modelUrl) {
    try {
      // 动态加载 pixi + live2d-display
      await this._loadLibs();
      const app = new PIXI.Application({
        transparent: true, resizeTo: window,
        autoStart: true, backgroundAlpha: 0,
      });
      // 挂到容器
      this.app = app;
      document.getElementById("live2d-root")?.appendChild(app.view);

      this.model = await PIXI.live2d.Live2DModel.from(modelUrl);
      app.stage.addChild(this.model);
      // 放大并居中
      const scale = Math.min(window.innerWidth, window.innerHeight) / 600;
      this.model.scale.set(scale * 2.5);
      this.model.anchor.set(0.5, 0.5);
      this.model.x = app.screen.width / 2;
      this.model.y = app.screen.height / 2;
      this.model.motion = "tap_body"; // 待机动作

      this.ready = true;
      if (this.fallback) this.fallback.stop();
      this._startLoop();
      console.log("[Live2D] model loaded:", modelUrl);
    } catch (e) {
      console.warn("[Live2D] load failed, using Canvas avatar:", e);
      this.ready = false;
      if (this.fallback) this.fallback.start();
    }
  }

  async _loadLibs() {
    if (window.PIXI) return;
    const script = (src) => new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
    // pixi.js + pixi-live2d-display
    await script("https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js");
    await script("https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js");
  }

  /** 喂入 int16 PCM bytes → RMS → 嘴开合 */
  feedAudio(bytes) {
    if (!bytes || bytes.length < 2) return;
    const n = bytes.length / 2;
    let sum = 0;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < n; i++) {
      const v = view.getInt16(i * 2, true) / 32768;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / n);
    this.smoothRMS = this.smoothRMS * 0.7 + rms * 0.3;
  }

  setState(s) {
    this.state = s;
    if (!this.ready || !this.model) {
      if (this.fallback) this.fallback.setState(s);
      return;
    }
    const { low, high, minOpenY, maxOpenY } = this.cfg;
    if (s === "speaking") {
      this.isTalking = true;
    } else if (s === "listening" || s === "thinking") {
      // 倾听/思考时嘴闭合，眼睛可微动
      this.model.internalModel.motionManager?.expressionManager?.setExpression?.("f01");
    } else {
      this.isTalking = false;
      this._setMouth(0);
      this.model.internalModel.motionManager?.expressionManager?.setExpression?.(null);
    }
    void low; void high; void minOpenY; void maxOpenY;
  }

  _setMouth(v) {
    if (!this.model || !this.model.internalModel) return;
    try {
      this.model.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", v);
      this.model.internalModel.coreModel.setParameterValueById("ParamMouthForm", v * 0.8);
      this.model.update();
    } catch (e) { /* 参数不存在则忽略 */ }
  }

  _startLoop() {
    if (this.raf) return;
    const loop = () => {
      // 音量包络 → 分段映射嘴开合
      if (this.isTalking) {
        const v = Math.min(1, this.smoothRMS / 0.12);
        // 非线性映射：低段缓，高段快
        let open;
        if (v < this.cfg.low) open = this.cfg.minOpenY + Math.pow(v / this.cfg.low, 1.5) * (0.4 - this.cfg.minOpenY);
        else if (v < this.cfg.high) open = 0.4 + ((v - this.cfg.low) / (this.cfg.high - this.cfg.low)) * 0.4;
        else open = 0.8 + Math.pow((v - this.cfg.high) / (1 - this.cfg.high), 1.2) * 0.2;
        this.mouthOpenY += (open - this.mouthOpenY) * 0.35;
        this._setMouth(Math.max(0, Math.min(1, this.mouthOpenY)));
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  destroy() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
    if (this.app) this.app.destroy(true);
    this.app = null;
  }
}

if (typeof window !== "undefined") window.Live2DAvatar = Live2DAvatar;

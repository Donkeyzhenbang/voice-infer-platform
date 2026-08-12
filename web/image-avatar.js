/**
 * 图片头像 — 加载角色图片 + 音频RMS驱动嘴部动画 + 说话弹跳
 *
 * 接口与 Canvas Avatar 兼容：
 *   const avatar = new ImageAvatar(canvas, { imageUrl, mouth })
 *   avatar.feedAudio(int16Bytes);
 *   avatar.setState('idle'|'speaking'|'thinking'|'listening');
 *
 * mouth 配置（相对图片的归一化坐标，可调）:
 *   { x: 0.5, y: 0.78, w: 0.2, h: 0.06 }  // 嘴中心x/y, 嘴宽/高比例
 */
class ImageAvatar {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.dpr = window.devicePixelRatio || 1;
    this.opts = Object.assign({
      imageUrl: "/assets/nailong/ref.png",
      mouth: { x: 0.5, y: 0.78, w: 0.18, h: 0.05 },
      bounce: true,        // 说话时轻微上下弹跳
      blink: false,        // 图片头像无眨眼（除非提供眼睛坐标）
    }, opts);
    this.state = "idle";
    this.img = new Image();
    this.loaded = false;
    this.smoothRMS = 0;
    this.mouthOpen = 0;
    this.mouthTarget = 0;
    this.bouncePhase = 0;
    this._raf = null;
    this.resize();
    this._load();
  }

  _load() {
    this.img.onload = () => {
      this.loaded = true;
      console.log("[ImageAvatar] loaded:", this.opts.imageUrl);
      if (this.state === "idle") this._ensureLoop();
    };
    this.img.onerror = (e) => console.warn("[ImageAvatar] image load failed:", this.opts.imageUrl, e);
    this.img.src = this.opts.imageUrl;
  }

  resize() {
    const w = this.cv.clientWidth || 320;
    const h = this.cv.clientHeight || 320;
    this.cv.width = w * this.dpr;
    this.cv.height = h * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.W = w; this.H = h;
  }

  setState(s) {
    this.state = s;
    if (s === "speaking") {
      this._ensureLoop();
    } else if (s === "idle") {
      this.mouthTarget = 0;
      if (this.loaded) this._ensureLoop();  // 保持待机微动
    } else {
      this.mouthTarget = 0;
    }
  }

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
    this.mouthTarget = Math.min(1, this.smoothRMS / 0.12);
    this._ensureLoop();
  }

  _ensureLoop() {
    if (this._raf || !this.loaded) return;
    const loop = () => {
      this.update();
      this.draw();
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  update() {
    this.mouthOpen += (this.mouthTarget - this.mouthOpen) * 0.3;
    if (this.state === "speaking" || this.state === "idle") this.bouncePhase += 0.15;
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    if (!this.loaded) {
      // 加载中显示占位
      ctx.fillStyle = "#e8f6ff";
      ctx.fillRect(0, 0, this.W, this.H);
      ctx.fillStyle = "#999";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("加载奶龙中...", this.W / 2, this.H / 2);
      return;
    }

    // 说话弹跳（±2%）
    let dy = 0;
    if (this.opts.bounce && this.state === "speaking") {
      dy = Math.sin(this.bouncePhase) * this.H * 0.015;
    }

    // 绘制图片，contain 适配 + 居中
    const iw = this.img.width, ih = this.img.height;
    const scale = Math.min(this.W / iw, this.H / ih);
    const dw = iw * scale, dh = ih * scale;
    const dx = (this.W - dw) / 2, dyi = (this.H - dh) / 2 + dy;

    ctx.save();
    // 圆形遮罩
    ctx.beginPath();
    ctx.arc(this.W / 2, this.H / 2, Math.min(this.W, this.H) / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(this.img, dx, dyi, dw, dh);

    // 嘴部动画：在图片嘴位置叠加开合的嘴
    if (this.mouthOpen > 0.05) {
      const m = this.opts.mouth;
      const mx = dx + m.x * dw;
      const my = dyi + m.y * dh;
      const mw = m.w * dw * (1 + this.mouthOpen * 0.6);
      const mh = m.h * dh * this.mouthOpen;
      ctx.fillStyle = "#4a2a1a";
      ctx.beginPath();
      ctx.ellipse(mx, my, mw / 2, Math.max(1, mh / 2), 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // 状态角标
    const labels = { idle: "😊 待机", speaking: "🗣 说话中", thinking: "🤔 思考中", listening: "👂 倾听中" };
    ctx.font = "13px sans-serif";
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.textAlign = "center";
    ctx.fillText(labels[this.state] || "", this.W / 2, this.H - 8);
  }
}

if (typeof window !== "undefined") window.ImageAvatar = ImageAvatar;

/**
 * Avatar: 原创蓝猫机器人（哆啦A梦致敬风）
 * Canvas 绘制 + 音频 RMS 驱动口型 + 眨眼动画
 *
 * 用法:
 *   const avatar = new Avatar(canvas);
 *   avatar.feedAudio(int16Bytes);  // 喂 TTS PCM
 *   avatar.setState('idle'|'speaking'|'thinking'|'listening');
 */
class Avatar {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.dpr = window.devicePixelRatio || 1;
    this.state = "idle";
    this.blink = 0;          // 0=睁开 1=半闭 2=全闭
    this.blinkTimer = 0;
    this.mouthOpen = 0;      // 0~1 嘴开合
    this.mouthTarget = 0;
    this.smoothRMS = 0;
    this._buf = [];          // 待处理音频帧
    this._analyser = null;
    this._raf = null;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const w = this.cv.clientWidth || 480;
    const h = this.cv.clientHeight || 480;
    this.cv.width = w * this.dpr;
    this.cv.height = h * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.W = w; this.H = h;
  }

  setState(s) { this.state = s; }

  /**
   * 喂入 int16 PCM bytes，计算 RMS 并平滑为嘴型开合度。
   * 简化：直接按字节块算 RMS，无需 AudioContext。
   */
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
    // 平滑包络
    this.smoothRMS = this.smoothRMS * 0.7 + rms * 0.3;
    // 映射到嘴开合：RMS 0~0.15 → 嘴 0~1
    this.mouthTarget = Math.min(1, this.smoothRMS / 0.12);
  }

  start() {
    if (this._raf) return;
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
    // 嘴型缓动
    this.mouthOpen += (this.mouthTarget - this.mouthOpen) * 0.25;
    // 眨眼（约每 3 秒一次，双帧）
    this.blinkTimer++;
    if (this.blinkTimer > 60 + Math.random() * 120) {
      this.blinkTimer = 0;
      this.blink = 2; // 全闭
    }
    if (this.blink > 0) this.blink -= 0.1;
  }

  /* ── 绘制蓝猫机器人 ────────────────────────────── */

  draw() {
    const ctx = this.ctx;
    const W = this.W, H = this.H;
    ctx.clearRect(0, 0, W, H);

    // 背景（浅蓝渐变）
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#bfe3ff");
    g.addColorStop(1, "#e8f6ff");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H * 0.55;
    const r = Math.min(W, H) * 0.32; // 头半径

    // ── 身体 ──
    ctx.fillStyle = "#29a5e8";
    this._roundRect(cx - r * 1.15, cy + r * 1.05, r * 2.3, r * 1.7, r * 0.5);
    ctx.fill();
    // 白肚皮 + 口袋
    ctx.fillStyle = "#ffffff";
    this._roundRect(cx - r * 0.95, cy + r * 1.15, r * 1.9, r * 1.4, r * 0.45);
    ctx.fill();
    // 口袋弧线
    ctx.strokeStyle = "#1976b8";
    ctx.lineWidth = r * 0.06;
    ctx.beginPath();
    ctx.arc(cx, cy + r * 1.55, r * 0.75, Math.PI * 0.05, Math.PI * 0.95);
    ctx.stroke();

    // ── 头（蓝色圆） ──
    ctx.fillStyle = "#29a5e8";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    // 白脸
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.ellipse(cx, cy + r * 0.12, r * 0.88, r * 0.82, 0, 0, Math.PI * 2);
    ctx.fill();

    // ── 眼睛（大圆，可眨） ──
    const eyeOpen = 1 - this.blink * 0.9;
    for (const side of [-1, 1]) {
      const ex = cx + side * r * 0.42;
      const ey = cy - r * 0.15;
      ctx.fillStyle = "#1a1a2e";
      ctx.beginPath();
      ctx.ellipse(ex, ey, r * 0.16, r * 0.22 * eyeOpen, 0, 0, Math.PI * 2);
      ctx.fill();
      if (eyeOpen > 0.5) {
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(ex - r * 0.05, ey - r * 0.06, r * 0.05, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ── 红鼻子（小圆） ──
    ctx.fillStyle = "#e63946";
    ctx.beginPath();
    ctx.arc(cx, cy + r * 0.18, r * 0.09, 0, Math.PI * 2);
    ctx.fill();

    // ── 胡须 ──
    ctx.strokeStyle = "#1a1a2e";
    ctx.lineWidth = r * 0.03;
    ctx.lineCap = "round";
    for (const side of [-1, 1]) {
      const sx = cx + side * r * 0.22;
      const sy = cy + r * 0.28;
      for (let i = -1; i <= 1; i++) {
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + side * r * 0.5, sy + i * r * 0.12);
        ctx.stroke();
      }
    }

    // ── 嘴（音量驱动开合） ──
    const mouthW = r * (0.3 + this.mouthOpen * 0.25);
    const mouthH = r * (0.05 + this.mouthOpen * 0.22);
    const my = cy + r * 0.38;
    ctx.fillStyle = "#5a2a1a";
    ctx.beginPath();
    if (this.mouthOpen < 0.12) {
      // 闭嘴微笑
      ctx.strokeStyle = "#5a2a1a";
      ctx.lineWidth = r * 0.05;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.arc(cx, my - r * 0.02, mouthW * 0.5, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
    } else {
      ctx.ellipse(cx, my, mouthW * 0.5, mouthH * 0.5, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── 状态角标 ──
    const labels = {
      idle: "😊 待机",
      speaking: "🗣 说话中",
      thinking: "🤔 思考中",
      listening: "👂 倾听中",
    };
    ctx.font = `${r * 0.22}px sans-serif`;
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.textAlign = "center";
    ctx.fillText(labels[this.state] || "", cx, H - r * 0.15);

    // 头顶红圈（哆啦A梦风天线）
    ctx.fillStyle = "#e63946";
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.95, r * 0.08, 0, Math.PI * 2);
    ctx.fill();
  }

  _roundRect(x, y, w, h, rad) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.lineTo(x + w - rad, y);
    ctx.arcTo(x + w, y, x + w, y + rad, rad);
    ctx.lineTo(x + w, y + h - rad);
    ctx.arcTo(x + w, y + h, x + w - rad, y + h, rad);
    ctx.lineTo(x + rad, y + h);
    ctx.arcTo(x, y + h, x, y + h - rad, rad);
    ctx.lineTo(x, y + rad);
    ctx.arcTo(x, y, x + rad, y, rad);
    ctx.closePath();
  }
}

// 浏览器全局导出
if (typeof window !== "undefined") window.Avatar = Avatar;

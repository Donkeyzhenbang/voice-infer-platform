/**
 * Emotes — 惊喜/反应表情贴纸（SVG 绘制，无图片依赖）
 * 触发后以浮层形式弹出，自动消失
 */
class Emotes {
  constructor(container) {
    this.container = container || document.body;
    this.defs = {
      surprise: { emoji: "😮", label: "哇！", color: "#ffe066" },
      happy: { emoji: "😄", label: "太棒了！", color: "#a8e6a3" },
      thinking: { emoji: "🤔", label: "让我想想", color: "#aad8ff" },
      love: { emoji: "😍", label: "爱了爱了", color: "#ffb3c6" },
      sad: { emoji: "😢", label: "别难过", color: "#b0b0c0" },
      sparkle: { emoji: "✨", label: "叮咚！", color: "#fff3a0" },
    };
  }

  /** 触发一个表情，如 show('surprise') */
  show(type) {
    const def = this.defs[type] || this.defs.surprise;
    const el = document.createElement("div");
    el.className = "emote-pop";
    el.innerHTML = `<div class="emote-emoji">${def.emoji}</div>
      <div class="emote-label">${def.label}</div>`;
    el.style.setProperty("--emote-color", def.color);
    this.container.appendChild(el);
    setTimeout(() => el.remove(), 1600);
  }

  /** 根据 LLM 情绪/关键词自动触发 */
  detect(text) {
    if (!text) return;
    const t = text;
    if (/惊喜|哇|天哪|竟然|真的吗/.test(t)) this.show("surprise");
    else if (/哈哈|太好了|开心|棒/.test(t)) this.show("happy");
    else if (/爱你|喜欢|可爱/.test(t)) this.show("love");
    else if (/唉|难过|伤心|可惜/.test(t)) this.show("sad");
    else if (/道具|口袋|铜锣烧|掏出/.test(t)) this.show("sparkle");
    else if (/想想|考虑|思考/.test(t)) this.show("thinking");
  }
}

if (typeof window !== "undefined") window.Emotes = Emotes;

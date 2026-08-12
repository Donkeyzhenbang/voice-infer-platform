from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from voice_infer.server.persona import load_persona, load_persona_registry
from voice_infer.server.session import SessionManager


def _config(default, entries):
    return SimpleNamespace(default=default, list=entries)


class PersonaTests(unittest.TestCase):
    def test_persona_registry_binds_voice_and_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            (assets / "ref.wav").write_bytes(b"RIFF-test")
            (assets / "ref.txt").write_text("逐字台词", encoding="utf-8")
            (root / "dora.md").write_text(
                "---\n"
                "name: 哆啦A梦\n"
                "voice_id: dora\n"
                "greeting: 你好呀\n"
                "ref_wav: assets/ref.wav\n"
                "ref_text: assets/ref.txt\n"
                "---\n"
                "你是来自未来的猫型机器人。\n",
                encoding="utf-8",
            )

            default_id, personas = load_persona_registry(
                root, _config("dora", {"dora": "dora.md"})
            )

            self.assertEqual(default_id, "dora")
            self.assertEqual(personas["dora"]["voice_id"], "dora")
            self.assertEqual(personas["dora"]["ref_wav"], str(assets / "ref.wav"))
            self.assertEqual(personas["dora"]["ref_text"], "逐字台词")
            self.assertEqual(personas["dora"]["text"], "你是来自未来的猫型机器人。")

    def test_missing_optional_voice_assets_fall_back(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dora.md").write_text(
                "---\nref_wav: missing.wav\nref_text: missing.txt\n---\n角色正文\n",
                encoding="utf-8",
            )
            _, personas = load_persona_registry(
                root, _config("dora", {"dora": "dora.md"})
            )
            self.assertIsNone(personas["dora"]["ref_wav"])
            self.assertEqual(personas["dora"]["ref_text"], "")

    def test_registry_rejects_unknown_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.md").write_text("人设", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "personas.default"):
                load_persona_registry(
                    root, _config("ghost", {"demo": "demo.md"})
                )

    def test_load_persona_keeps_greeting_and_voice(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.md"
            path.write_text(
                "---\nid: robot\nvoice_id: warm\ngreeting: 欢迎回来\n---\n正文\n",
                encoding="utf-8",
            )
            persona = load_persona(path)
            self.assertEqual(persona["id"], "robot")
            self.assertEqual(persona["voice_id"], "warm")
            self.assertEqual(persona["greeting"], "欢迎回来")

    def test_persona_switch_updates_voice_atomically(self):
        sessions = SessionManager()
        session = sessions.create("sid", "default", "旧人设", "default")
        sessions.update_persona("sid", "dora", "新人设", "dora")
        self.assertEqual(session.persona_id, "dora")
        self.assertEqual(session.persona_text, "新人设")
        self.assertEqual(session.voice_id, "dora")


if __name__ == "__main__":
    unittest.main()

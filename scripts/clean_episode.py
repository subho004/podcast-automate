#!/usr/bin/env python3
import sys
import json
from pathlib import Path

def strip_code_fence(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("\n", 1)
        if len(parts) == 2:
            s = parts[1]
    if s.endswith("```"):
        s = s[:-3]
    s = s.replace("```json", "").replace("```", "")
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def extract_json(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None


def clean_scene(scene: dict):
    maya_text = scene.get("maya_text") or ""
    llm_raw = scene.get("llm_raw") or ""

    # Prefer parsing llm_raw if available
    parsed = extract_json(strip_code_fence(llm_raw)) or extract_json(llm_raw)
    if parsed:
        text = parsed.get("text")
        voice = parsed.get("voice_description") or parsed.get("voice")
        if text:
            scene["maya_text"] = strip_code_fence(text)
        else:
            scene["maya_text"] = strip_code_fence(maya_text)
        if voice:
            scene["voice_description"] = strip_code_fence(voice)
        else:
            scene["voice_description"] = strip_code_fence(scene.get("voice_description") or "")
    else:
        # fallback: just strip fences from existing fields
        scene["maya_text"] = strip_code_fence(maya_text)
        scene["voice_description"] = strip_code_fence(scene.get("voice_description") or "")
    # remove legacy/verbose fields
    scene.pop("blurb", None)
    scene.pop("llm_raw", None)
    scene.pop("maya_description", None)
    return scene


def main(path: str):
    p = Path(path)
    if not p.exists():
        print("File not found:", p)
        sys.exit(2)
    with p.open("r", encoding="utf-8") as f:
        ep = json.load(f)

    for s in ep.get("scenes", []):
        clean_scene(s)

    with p.open("w", encoding="utf-8") as f:
        json.dump(ep, f, indent=2, ensure_ascii=False)

    print("Cleaned:", p)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: clean_episode.py <path/to/episode.json>")
        sys.exit(1)
    main(sys.argv[1])

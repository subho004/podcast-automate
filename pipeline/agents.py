import json
from typing import List
from . import llm


def generate_episode(query: str, scenes: int | None = 3) -> dict:
    """Generate a simple episode structure using the LLM outline.

    Returns an episode dict with scene list containing id, title, blurb.
    """
    text = llm.generate_outline(query, scenes=scenes)
    try:
        data = json.loads(text)
        # If model returned scenes_count but not scenes, guard
        if "scenes" not in data and data.get("scenes_count"):
            # no scenes returned, create placeholders
            n = int(data.get("scenes_count"))
            data["scenes"] = [{"id": i + 1, "title": f"Scene {i+1}", "blurb": ""} for i in range(n)]
    except Exception:
        # fallback: try to parse lines
        data = {"title": query, "scenes": []}
        lines = text.splitlines()
        max_lines = scenes if scenes is not None else min(5, max(3, len(lines)))
        for i, line in enumerate(lines[:max_lines]):
            data["scenes"].append({"id": i + 1, "title": f"Scene {i+1}", "blurb": line.strip()})
    return data

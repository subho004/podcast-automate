#!/usr/bin/env python3
"""Simple test script for the podcast-automate pipeline.

Usage:
  # Dry run (LLM outline + scene->Maya conversion only):
  python scripts/test_pipeline.py "A cozy sci-fi short about an AI barista"

  # Full run (synthesis + merging) -- can be slow and download models:
  python scripts/test_pipeline.py "A cozy sci-fi short about an AI barista" --full --scenes 2 --voice "warm conversational male"
"""
import argparse
import os
import textwrap
import time
import sys
from pathlib import Path

# Ensure project root is on sys.path so local packages (pipeline, audio, etc.) can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import agents, llm


def dry_run(query: str, scenes: int | None, voice: str):
    print("DRY RUN: generating outline via LLM")
    outline_text = agents.generate_episode(query, scenes=scenes)
    # agents.generate_episode may return dict or text; handle both
    if isinstance(outline_text, str):
        print("LLM produced (raw):\n", outline_text)
        return

    episode = outline_text
    print("Episode title:", episode.get("title"))
    for s in episode.get("scenes", []):
        print("---")
        print(f"Scene {s.get('id')}: {s.get('title')}")
        print("Converting scene -> Maya text via OpenAI...")
        maya = llm.scene_to_maya(s, voice_hint=voice)
        print("voice_description:", maya.get("voice_description"))
        print("maya text preview:", textwrap.shorten(maya.get("text", ""), width=300))


def full_run(query: str, scenes: int | None, voice: str):
    print("FULL RUN: starting orchestrator (this may download models and take several minutes)")
    from orchestrator import run as orchestrator_run
    start = time.perf_counter()
    orchestrator_run(query, scenes=scenes, voice_description=voice)
    print("Finished. Check outputs/ for results.")
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", help="User query for the episode")
    p.add_argument("--scenes", type=int, default=None, help="Number of scenes; omit to let LLM choose")
    p.add_argument("--voice", type=str, default=None)
    p.add_argument("--full", action="store_true", help="Run full synthesis (may download models)")
    args = p.parse_args()

    if not args.full:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set in environment. Export it or set in .env before running dry-run.")
            return
        dry_run(args.query, args.scenes, args.voice)
    else:
        full_run(args.query, args.scenes, args.voice)


if __name__ == "__main__":
    main()

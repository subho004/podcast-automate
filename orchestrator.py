import argparse
import json
import uuid
from pathlib import Path
import time
import traceback
import logging
from datetime import datetime

from pipeline import agents
from pipeline.llm import scene_to_maya, validate_maya_text
from audio.generator import SimpleAudioGenerator, PersistentAudioGenerator
from audio.merge import concat_wavs
import re

# Configure basic logging for the orchestrator module
logger = logging.getLogger("podcast_orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run(query: str, scenes: int | None = None, voice_description: str = None):
    episode = agents.generate_episode(query, scenes=scenes)
    episode_id = str(uuid.uuid4())[:8]
    ep_dir = Path("outputs") / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Use persistent generator to avoid reloading models for each scene
    generator = PersistentAudioGenerator(out_dir="outputs")
    scene_files = []
    start_time = time.perf_counter()
    total_scenes = len(episode.get("scenes", []))
    logger.info(f"Starting episode generation: planned scenes={total_scenes}")
    synthesized = 0
    failed = 0
    scene_stats = []
    log_path = ep_dir / "maya_log.jsonl"
    def _retry_call(fn, name: str, retries: int = 3, delay: float = 1.0, validate=None):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                res = fn()
                if validate and not validate(res):
                    raise ValueError("validation failed")
                return res
            except Exception as e:
                last_exc = e
                print(f"[{name}] attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(delay * (2 ** (attempt - 1)))
        print(f"[{name}] all {retries} attempts failed")
        traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)
        raise last_exc

    for i, s in enumerate(episode.get("scenes", []), start=1):
        logger.info(f"Preparing scene {i}/{total_scenes}: {s.get('title')}")

        # Convert scene blurb to Maya text + voice description via LLM with retries
        try:
            maya = _retry_call(
                lambda: scene_to_maya(s, voice_hint=voice_description),
                name=f"scene_to_maya.scene_{i}",
                retries=3,
                delay=1.0,
                validate=lambda r: bool(r and r.get("text")),
            )
        except Exception:
            logger.warning(f"Skipping scene {i} due to LLM conversion failure")
            s["error"] = "llm_conversion_failed"
            failed += 1
            continue

        scene_text = maya.get("text")
        # normalize voice description to a single-line, compact string
        raw_voice = maya.get("voice_description") or voice_description or "Neutral conversational voice, warm timbre."
        scene_voice = re.sub(r"\s+", " ", raw_voice).strip()

        # record LLM conversion time and write a pre-synthesis log entry
        pre_ts = datetime.utcnow().isoformat() + "Z"
        # store only the cleaned voice_description and final maya_text.
        s["maya_text"] = scene_text
        s["voice_description"] = scene_voice

        # write a pre-synthesis log entry (full maya_text included)
        try:
            pre_entry = {
                "ts": pre_ts,
                "scene": i,
                "title": s.get("title"),
                "voice_description": scene_voice,
                "maya_text": scene_text,
                "stage": "before_synthesis",
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(pre_entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Could not write pre-synthesis log for scene %s", i, exc_info=True)

        # Validate Maya emotion tags and sanitize unknown tags
        try:
            valid, sanitized, unknown = validate_maya_text(scene_text)
            if not valid:
                logger.warning(f"Scene {i}: unknown emotion tags found and removed: {unknown}")
                s["tags_issues"] = list(unknown)
                scene_text = sanitized
        except Exception:
            logger.debug("Error validating maya text for scene %s", i, exc_info=True)
        # Log the Maya-formatted text (trimmed) for auditing
        try:
            preview = (scene_text or "").replace("\n", " ")
            logger.info(f"Scene {i}/{total_scenes} maya_text preview: {preview[:400]}{'' if len(preview) <= 400 else '...' }")
        except Exception:
            logger.debug("Could not log maya_text for scene %s", i, exc_info=True)

        logger.info(f"Synthesizing scene {i}/{total_scenes}: {s.get('title')}")
        try:
            # use the single-line voice_description for the <description="..."> insert as recommended by Maya1 docs
            res = _retry_call(
                # pass only the single-line `scene_voice` to the synthesizer
                lambda: generator.synthesize_scene(scene_text, scene_voice, episode_id, i),
                name=f"synthesize.scene_{i}",
                retries=2,
                delay=2.0,
                validate=lambda r: bool(r and r.get("audio")),
            )
        except Exception as synth_exc:
            logger.warning(f"Synthesis failed for scene {i}; marking failed and continuing")
            s["error"] = "synthesis_failed"
            failed += 1
            # write a failure log entry
            try:
                fail_entry = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "scene": i,
                    "title": s.get("title"),
                    "stage": "synthesis_failed",
                    "error": str(synth_exc),
                }
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
            except Exception:
                logger.debug("Could not write failure log for scene %s", i, exc_info=True)
            continue

        scene_files.append(res["audio"])
        s["audio_file"] = res["audio"]
        s["duration"] = res["duration"]
        synthesized += 1
        timings = res.get("timings", {})
        gen_t = timings.get("generation")
        dec_t = timings.get("decoding")
        scene_stats.append({"scene": i, "gen_time": gen_t, "decode_time": dec_t, "duration": s.get("duration")})
        logger.info(f"Completed scene {i}/{total_scenes}: audio={res.get('audio')}, duration={s.get('duration'):.2f}s, gen={gen_t:.2f}s, decode={dec_t:.2f}s")
        # Also log the maya_text again on completion (short preview)
        try:
            preview = (s.get("maya_text") or "").replace("\n", " ")
            logger.info(f"Scene {i}/{total_scenes} maya_text final preview: {preview[:400]}{'' if len(preview) <= 400 else '...' }")
        except Exception:
            logger.debug("Could not log final maya_text for scene %s", i, exc_info=True)

        # write a post-synthesis log entry
        try:
            post_entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "scene": i,
                "title": s.get("title"),
                "stage": "synthesized",
                "audio": res.get("audio"),
                "duration": s.get("duration"),
                "timings": timings,
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(post_entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Could not write post-synthesis log for scene %s", i, exc_info=True)

    # strip legacy fields from scenes before saving
    for sc in episode.get("scenes", []):
        sc.pop("blurb", None)
        sc.pop("llm_raw", None)
        sc.pop("maya_description", None)

    final_path = ep_dir / "episode.wav"
    concat_wavs(scene_files, str(final_path))
    episode["id"] = episode_id
    episode["total_audio"] = str(final_path)
    with open(ep_dir / "episode.json", "w") as f:
        json.dump(episode, f, indent=2)

    wall_elapsed = time.perf_counter() - start_time
    total_audio_duration = sum(s.get("duration", 0) for s in episode.get("scenes", []))
    logger.info(f"Episode ready: {final_path}")
    logger.info(f"Episode summary: id={episode_id}, planned={total_scenes}, synthesized={synthesized}, failed={failed}, total_audio={total_audio_duration:.2f}s, wall_time={wall_elapsed:.2f}s")
    logger.info("Per-scene stats:")
    for st in scene_stats:
        logger.info(f" scene {st['scene']}: gen={st['gen_time']:.2f}s decode={st['decode_time']:.2f}s audio_duration={st['duration']:.2f}s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Generate a new episode from a query")
    run_p.add_argument("query")
    run_p.add_argument("--scenes", type=int, default=None, help="Number of scenes; omit to let LLM choose")
    run_p.add_argument("--voice", type=str, default=None)

    add_p = sub.add_parser("add-scene", help="Add a scene to an existing episode")
    add_p.add_argument("episode_id", help="Episode ID (folder under outputs)")
    add_p.add_argument("--text", required=True, help="Scene text/blurb to synthesize")
    add_p.add_argument("--title", default=None, help="Optional scene title")
    add_p.add_argument("--voice", default=None, help="Optional voice description for this scene")

    args = p.parse_args()

    if args.cmd == "run":
        run(args.query, scenes=args.scenes, voice_description=args.voice)
    elif args.cmd == "add-scene":
        add_scene(args.episode_id, args.text, title=args.title, voice_description=args.voice)


def add_scene(episode_id: str, text: str, title: str = None, voice_description: str = None):
    """Append a new scene to an existing episode, synthesize it, and rebuild the final episode audio."""
    ep_dir = Path("outputs") / episode_id
    meta_path = ep_dir / "episode.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Episode metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        episode = json.load(f)

    scenes = episode.get("scenes", [])
    next_idx = len(scenes) + 1
    scene_title = title or f"Scene {next_idx}"
    log_path = ep_dir / "maya_log.jsonl"

    print(f"Adding scene {next_idx} to episode {episode_id}: {scene_title}")

    def _retry_call(fn, name: str, retries: int = 3, delay: float = 1.0, validate=None):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                res = fn()
                if validate and not validate(res):
                    raise ValueError("validation failed")
                return res
            except Exception as e:
                last_exc = e
                print(f"[{name}] attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(delay * (2 ** (attempt - 1)))
        print(f"[{name}] all {retries} attempts failed")
        traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)
        raise last_exc

    # Convert new scene to Maya text via LLM with retries
    try:
        maya = _retry_call(
            lambda: scene_to_maya({"title": scene_title, "blurb": text}, voice_hint=voice_description),
            name=f"scene_to_maya.add_scene_{next_idx}",
            retries=3,
            delay=1.0,
            validate=lambda r: bool(r and r.get("text")),
        )
    except Exception:
        raise RuntimeError("Failed to generate Maya prompt for new scene")

    scene_text = maya.get("text")
    raw_voice = maya.get("voice_description") or voice_description or "Neutral conversational voice, warm timbre."
    scene_voice = re.sub(r"\s+", " ", raw_voice).strip()
    maya_desc = scene_voice

    # write pre-synthesis log
    try:
        pre_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "scene": next_idx,
            "title": scene_title,
            "voice_description": scene_voice,
            "maya_text": scene_text,
            "stage": "before_synthesis",
        }
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(pre_entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Could not write pre-synthesis log for add_scene %s", next_idx, exc_info=True)

    generator = PersistentAudioGenerator(out_dir="outputs")
    try:
        res = _retry_call(
            lambda: generator.synthesize_scene(scene_text, scene_voice, episode_id, next_idx),
            name=f"synthesize.add_scene_{next_idx}",
            retries=2,
            delay=2.0,
            validate=lambda r: bool(r and r.get("audio")),
        )
    except Exception as synth_exc:
        # write failure entry
        try:
            fail_entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "scene": next_idx,
                "title": scene_title,
                "stage": "synthesis_failed",
                "error": str(synth_exc),
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(fail_entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Could not write failure log for add_scene %s", next_idx, exc_info=True)
        raise RuntimeError("Failed to synthesize audio for new scene")

    new_scene = {
        "id": next_idx,
        "title": scene_title,
        "maya_text": scene_text,
        "voice_description": scene_voice,
        "audio_file": res["audio"],
        "duration": res.get("duration"),
    }
    scenes.append(new_scene)
    episode["scenes"] = scenes

    # write post-synthesis entry for the added scene
    try:
        post_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "scene": next_idx,
            "title": scene_title,
            "stage": "synthesized",
            "audio": res.get("audio"),
            "duration": res.get("duration"),
        }
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(post_entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Could not write post-synthesis log for add_scene %s", next_idx, exc_info=True)

    # rebuild final episode audio
    scene_files = [s["audio_file"] for s in scenes if s.get("audio_file")]
    final_path = ep_dir / "episode.wav"
    concat_wavs(scene_files, str(final_path))
    episode["total_audio"] = str(final_path)

    with open(meta_path, "w") as f:
        json.dump(episode, f, indent=2)

    print(f"Scene added. Updated episode at {meta_path}")


if __name__ == "__main__":
    main()

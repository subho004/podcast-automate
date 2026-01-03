# podcast-automate

Minimal prototype to generate short podcast episodes from a text prompt using an LLM + Maya1 TTS pipeline.

## Demo

Listen to a generated demo episode (relative path). If you upload this repository to GitHub and include the `outputs/624d331d/episode.wav` file, the player below will let visitors listen directly from the README:

<audio controls>
  <source src="outputs/624d331d/episode.wav" type="audio/wav">
  Your browser does not support the audio element. You can also download the file from [outputs/624d331d/episode.wav](outputs/624d331d/episode.wav).
</audio>

Prompt used to create this demo: "A cozy sci-fi short about an AI barista"

## Overview

This repo contains a small FastAPI app, an orchestrator CLI, and a pipeline that chains: agent prompts → OpenAI LLM (scene outlines) → Maya1 TTS → SNAC decoding → per-scene WAV files → final merged episode.

Key pieces

- FastAPI endpoints under `app/` (dev server: `uvicorn app.main:app --reload`)
- Orchestrator CLI and background runner: `orchestrator.py`
- Persistent TTS loader & audio merge in `audio/`
- LLM adapters in `pipeline/`

## Notes about development machine vs recommended hardware

- Current development and testing were performed on a MacBook (Apple Silicon). That works well for light experimentation and small demos.
- For faster end-to-end runs and comfortable local development when working with larger models, we recommend stronger hardware:
  - CPU: 6+ physical cores (Intel i7/i9 or Apple M1 Pro/Max / M2 Pro/Max)
  - RAM: 32 GB+ for multi-model workflows; 16 GB is workable for small tests
  - GPU: A discrete CUDA GPU (NVIDIA with 8+ GB VRAM) or Apple Silicon (M1/M2) with sufficient unified memory speeds up synthesis significantly
  - Disk: 50+ GB free to cache model weights and artifacts (models vary in size)

You can still run the pipeline on modest hardware, but expect longer inference times and potential memory constraints for large models.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI key (either export or create a `.env`):

```bash
export OPENAI_API_KEY="sk-..."
# or create a .env with OPENAI_API_KEY=...
```

## Running the API

```bash
uvicorn app.main:app --reload
```

## Orchestrator / CLI

Generate an episode via the CLI. Example (request 3 scenes and a warm female voice):

```bash
python orchestrator.py "A cozy sci-fi short about an AI barista" --scenes 3 --voice "warm conversational female"
```

## LLM scene-count and customization

- The LLM prompt (pipeline/llm.py) currently requests concise scene lines and suggests producing between 3–6 short scenes by default. This can be changed to suit the episode length you want.
- You can control scene count from the CLI with the `--scenes` flag (or by editing the default/variable inside `pipeline/llm.py` if you prefer a code-level change).

## Test runner (smoke / full)

Quick smoke-run (no long model downloads if already cached):

```bash
python scripts/test_pipeline.py "A cozy sci-fi short about an AI barista" --voice "warm conversational female"
```

Full run (end-to-end, may download models and take several minutes):

```bash
python scripts/test_pipeline.py "A cozy sci-fi short about an AI barista" --full --voice "warm conversational female"
```

## API / frontend suggestions

- The repo includes a FastAPI app and an orchestrator that already supports background task-style episode creation with per-scene audit logs and outputs. The current flow writes artifacts to `outputs/<episode_id>/` and exposes progress via `maya_log.jsonl` and `episode.json`.
- Typical production architecture: expose an endpoint that enqueues a task (returns a job id), poll a status endpoint (or use websockets) for progress, and then fetch the final `episode.wav` (or stream it) when ready. The existing code already implements the core pieces you need to build this pattern — a frontend can easily be added to create episodes, show progress per scene, and play the final audio.

## Outputs & logs

- Generated artifacts are written to `outputs/<episode_id>/`.
- Important files: `episode.wav` (final mix), `episode.json` (metadata), `maya_log.jsonl` (per-scene audit lines).

## Troubleshooting

- If you run into native wheel / PyTorch installation issues, use the PyTorch install selector at https://pytorch.org/get-started/locally/ to pick the correct `pip` command for your platform.
- The first run downloads model weights — expect multiple minutes on a cold machine.
- If you see audio errors related to `audioop`/`pyaudioop`, ensure `soundfile` and `numpy` are installed (they are in `requirements.txt`) — this project avoids `pydub` native deps.

## Maya1 TTS Documentation

This project uses Maya1 for high-quality text-to-speech synthesis in the podcast generation pipeline.

Model & documentation: https://huggingface.co/maya-research/maya1

Purpose in this repo: Converts per-scene LLM output into natural-sounding speech, which is later decoded and merged into a full podcast episode.

Notes:

- The first run may download large model weights.
- Performance depends on available CPU/GPU and system memory.
- Refer to the Hugging Face page for licensing, usage examples, and hardware recommendations.
- For any Maya1-specific configuration or model updates, always consult the official documentation linked above.

## Contributing

Contributions are welcome. Please follow these guidelines to make collaboration smooth:

- Fork the repository and create a feature branch for your work.
- Open a pull request with a clear title and description summarizing the change.
- Follow the existing code style and keep changes focused and testable.
- Add or update tests where appropriate and include short usage notes in your PR.
- For large changes, open an issue first to discuss the design.

## License

This project is provided under the MIT License by default. Add a `LICENSE` file to the repository root with the exact license text to make it explicit. If you prefer a different license, tell me which one and I will add it.

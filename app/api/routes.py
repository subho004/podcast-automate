from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import uuid
import json

from orchestrator import run as orchestrator_run, add_scene as orchestrator_add_scene

router = APIRouter()


class CreateEpisodeRequest(BaseModel):
    query: str
    scenes: Optional[int] = 3
    voice: Optional[str] = None


class AddSceneRequest(BaseModel):
    text: str
    title: Optional[str] = None
    voice: Optional[str] = None


@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.post("/episodes", status_code=202)
async def create_episode(req: CreateEpisodeRequest, background_tasks: BackgroundTasks):
    """Start creating an episode in background. Returns a temporary job id (episode id after created)."""
    job_id = str(uuid.uuid4())[:8]
    # run orchestrator in background; it creates its own episode id, but we pass through a wrapper
    def _job():
        orchestrator_run(req.query, scenes=req.scenes, voice_description=req.voice)

    background_tasks.add_task(_job)
    return {"job_id": job_id, "status": "started"}


@router.post("/episodes/{episode_id}/scenes", status_code=202)
async def api_add_scene(episode_id: str, req: AddSceneRequest, background_tasks: BackgroundTasks):
    """Add a scene to an existing episode in background."""
    ep_dir = Path("outputs") / episode_id
    if not ep_dir.exists():
        raise HTTPException(status_code=404, detail="episode not found")

    def _job():
        orchestrator_add_scene(episode_id, req.text, title=req.title, voice_description=req.voice)

    background_tasks.add_task(_job)
    return {"episode_id": episode_id, "status": "add-scene started"}


@router.get("/episodes", response_model=List[str])
async def list_episodes():
    base = Path("outputs")
    if not base.exists():
        return []
    return [p.name for p in base.iterdir() if p.is_dir()]


@router.get("/episodes/{episode_id}")
async def get_episode(episode_id: str):
    meta = Path("outputs") / episode_id / "episode.json"
    if not meta.exists():
        raise HTTPException(status_code=404, detail="episode not found")
    return json.loads(meta.read_text())


@router.get("/episodes/{episode_id}/download")
async def download_episode(episode_id: str):
    episode_file = Path("outputs") / episode_id / "episode.wav"
    if not episode_file.exists():
        raise HTTPException(status_code=404, detail="episode audio not found")
    return {"path": str(episode_file)}

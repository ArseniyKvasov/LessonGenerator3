from __future__ import annotations

import logging
from typing import Any, Optional
from datetime import datetime, timezone
import threading
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.clients.groq import models_available
from app.generators.brief import generate_brief
from app.generators.tasks import generate_tasks
from config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lesson Generator API")

Job = dict[str, Any]
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


class BriefGenerateRequest(BaseModel):
    user_request: str = Field(..., min_length=1, max_length=3000)


class TasksGenerateRequest(BaseModel):
    brief: dict[str, Any]


@app.get("/health/")
def health() -> dict[str, Any]:
    return {"status": "ok", "models_available": models_available()}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY is not configured.",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


def _create_job(job_type: str) -> str:
    job_id = uuid4().hex
    now = _utc_now()

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

    logger.info("Created %s generation job %s", job_type, job_id)
    return job_id


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return

        job.update(fields)
        job["updated_at"] = _utc_now()


def _get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _run_job(job_id: str, generator, failure_message: str, *args: Any) -> None:
    _update_job(job_id, status="running")
    logger.info("Started generation job %s", job_id)

    try:
        result = generator(*args)
    except Exception as exc:
        _update_job(job_id, status="failed", error=f"{failure_message}: {exc}")
        logger.exception("Generation job %s failed with an exception", job_id)
        return

    if result.get("status") != "ok":
        error = result.get("error", failure_message)
        _update_job(
            job_id,
            status="failed",
            error=error,
        )
        logger.error("Generation job %s failed: %s", job_id, error)
        return

    _update_job(job_id, status="succeeded", result=result)
    logger.info("Generation job %s succeeded", job_id)


@app.post("/generate/brief/", status_code=status.HTTP_202_ACCEPTED)
def generate_lesson_brief(
    payload: BriefGenerateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict[str, str]:
    _require_api_key(x_api_key)
    job_id = _create_job("brief")
    background_tasks.add_task(
        _run_job,
        job_id,
        generate_brief,
        "Brief generation failed.",
        payload.user_request,
    )

    return {"job_id": job_id, "status": "queued"}


@app.post("/generate/tasks/", status_code=status.HTTP_202_ACCEPTED)
def generate_lesson_tasks(
    payload: TasksGenerateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict[str, str]:
    _require_api_key(x_api_key)
    job_id = _create_job("tasks")
    background_tasks.add_task(
        _run_job,
        job_id,
        generate_tasks,
        "Task generation failed.",
        payload.brief,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}/")
def get_job(job_id: str) -> Job:
    job = _get_job(job_id)

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    return job

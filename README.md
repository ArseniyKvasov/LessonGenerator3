# Lesson Generator API

FastAPI service for generating ESL lesson briefs and lesson tasks.

Generation is asynchronous: create a job, then poll it by `job_id`.

## Endpoints

- `GET /health/` - health check with model availability.
- `POST /generate/brief/` - creates a lesson brief generation job.
- `POST /generate/tasks/` - creates a lesson tasks generation job.
- `GET /jobs/{job_id}/` - returns job status and result.

## Environment

Copy `.env.example` to `.env` and fill in the keys:

```bash
cp .env.example .env
```

Required:

- `API_KEY` - service API key. Required in `X-API-Key` for generation endpoints.
- `GROQ_API_KEY` - text generation.
- `POLLINATIONS_API_KEY` - image and audio generation.

Pollinations defaults are set for low resource usage:

- Images: `gptimage`, `512x512`, `low`.
- Audio: `qwen-tts`, `mp3`.

You can override them with:

- `POLLINATIONS_IMAGE_MODEL`
- `POLLINATIONS_IMAGE_SIZE`
- `POLLINATIONS_IMAGE_QUALITY`
- `POLLINATIONS_AUDIO_MODEL`
- `POLLINATIONS_AUDIO_VOICE`
- `POLLINATIONS_AUDIO_FORMAT`

## Health And Rate Limits

`GET /health/` returns:

```json
{
  "status": "ok",
  "models_available": true
}
```

The service tracks Groq rate limits per model. When Groq returns HTTP `429`, the service reads the `retry-after` response header, marks only that model unavailable until the retry time, and tries another available model. If every Groq model is currently unavailable, `models_available` becomes `false`.

Jobs are stored in memory. For multiple production workers or restarts, use an external queue/storage such as Redis, Postgres, Celery, or RQ.

## Local Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker build -t lesson-generator-api .
docker run --env-file .env -p 8000:8000 lesson-generator-api
```

## Docker Compose

```bash
docker compose up --build
```

## Examples

Generate a brief job:

```bash
curl -X POST http://127.0.0.1:8000/generate/brief/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"user_request":"Business English lesson about meetings for B1 student"}'
```

Poll the job:

```bash
curl http://127.0.0.1:8000/jobs/{job_id}/
```

Generate tasks job:

```bash
curl -X POST http://127.0.0.1:8000/generate/tasks/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "brief": {
      "topic": "Business Meetings",
      "lesson_description": "Practice useful meeting vocabulary and speaking.",
      "sections": {
        "vocabulary": ["agenda", "deadline", "follow up", "minutes"],
        "grammar": "",
        "reading": "",
        "listening": "Short dialogue between colleagues planning a meeting.",
        "speaking": "Discuss a workplace meeting image."
      }
    }
  }'
```

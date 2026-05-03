import json
from typing import Any, Optional

from app.clients.groq import call_ai


MAX_USER_REQUEST_LENGTH = 1000
MAX_ATTEMPTS = 3


def build_brief_prompt(user_request: str, previous_error: Optional[str] = None) -> str:
    payload = {
        "user_request": user_request,
        "task": (
            "Process the user's request and create a lesson plan "
            "for a personal English as a Second Language lesson."
        ),
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown.",
            "Do not add explanations outside JSON.",
            "If the user provided only a topic, suggest suitable vocabulary or grammar depending on the topic.",
            "If the user provided a detailed lesson description, use it to create the plan.",
            "Fill topic first.",
            "topic must be a short lesson title, 1-3 words.",
            "Fill lesson_description after topic.",
            "lesson_description must be a short description for the tutor explaining what the lesson will include.",
            "Fill sections based on the lesson goal.",
            "Sections (vocabulary, grammar, reading, listening, speaking) are optional. Include only those that fit the lesson focus. Do not add a section just to follow a template.",
            "If a section is not needed, return an empty string or an array with one empty string.",
            "Do not force vocabulary if the lesson is focused only on grammar.",
            "Do not force grammar if the lesson is focused only on vocabulary.",
            "Add 1 or 2 practical sections: reading, listening, speaking.",
            "If it's a vocabulary-based lesson, vocabulary should include a list of all words and phrases.",
            "Consider it is a personal lesson.",
            "If the speaking section is included, do not add role-play or pair work. It must include personal and general questions."
        ],
        "response_schema": {
            "topic": "string",
            "lesson_description": "string",
            "sections": {
                "vocabulary": ["string"],
                "grammar": "string",
                "reading": "string",
                "listening": "string",
                "speaking": "string",
            },
        },
    }

    if previous_error:
        payload["previous_error"] = previous_error
        payload["fix_rules"] = [
            "Fix the previous error.",
            "Return only valid JSON matching the schema.",
        ]

    return json.dumps(payload, ensure_ascii=False)


def parse_ai_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {
            "status": "error",
            "error": "AI response must be a dictionary.",
        }

    if response.get("status") != "ok":
        return {
            "status": "error",
            "error": "AI response status is not ok.",
        }

    raw_content = response.get("response")

    if isinstance(raw_content, dict):
        brief = raw_content
    elif isinstance(raw_content, str):
        try:
            brief = json.loads(raw_content)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "error": "AI response is not valid JSON.",
            }
    else:
        return {
            "status": "error",
            "error": "AI response field must be JSON string or dictionary.",
        }

    validation_error = validate_brief(brief)
    if validation_error:
        return {
            "status": "error",
            "error": validation_error,
        }

    return {
        "status": "ok",
        "brief": brief,
    }


def has_filled_section(sections: dict[str, Any]) -> bool:
    for value in sections.values():
        if isinstance(value, str) and value.strip():
            return True

        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return True

    return False


def validate_brief(brief: Any) -> Optional[str]:
    if not isinstance(brief, dict):
        return "Brief must be a dictionary."

    topic = brief.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        return "topic is required."

    topic_words = topic.strip().split()
    if len(topic_words) > 3:
        return "topic must be 1-3 words."

    lesson_description = brief.get("lesson_description")
    if not isinstance(lesson_description, str) or not lesson_description.strip():
        return "lesson_description is required."

    sections = brief.get("sections")
    if not isinstance(sections, dict):
        return "sections is required and must be a dictionary."

    required_sections = ["vocabulary", "grammar", "reading", "listening", "speaking"]

    for section_name in required_sections:
        if section_name not in sections:
            return f"Missing section: {section_name}."

    vocabulary = sections.get("vocabulary")
    if not isinstance(vocabulary, list):
        return "sections.vocabulary must be a list."

    for item in vocabulary:
        if not isinstance(item, str):
            return "Every vocabulary item must be a string."

    for section_name in ["grammar", "reading", "listening", "speaking"]:
        if not isinstance(sections.get(section_name), str):
            return f"sections.{section_name} must be a string."

    if not has_filled_section(sections):
        return "At least one section must be filled."

    return None


def generate_brief(user_request: str) -> dict[str, Any]:
    user_request = user_request[:MAX_USER_REQUEST_LENGTH]
    previous_error = None

    for _ in range(MAX_ATTEMPTS):
        prompt = build_brief_prompt(user_request, previous_error)
        response = call_ai(prompt)
        result = parse_ai_response(response)

        if result["status"] == "ok":
            return {
                "status": "ok",
                "brief": result["brief"],
            }

        previous_error = result["error"]

    return {
        "status": "error",
        "error": previous_error or "Failed to generate lesson brief.",
    }
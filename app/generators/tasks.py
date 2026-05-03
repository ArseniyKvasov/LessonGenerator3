import asyncio
import json
import logging
from typing import Any, Optional

from app.clients.groq import call_ai
from app.clients.pollinations import generate_audio, generate_image

MAX_ATTEMPTS = 3
MEDIA_TIMEOUT = 60
LOG_PAYLOAD_PREVIEW_LENGTH = 1000

logger = logging.getLogger(__name__)


def _dump_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_ai_json(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"status": "error", "error": "AI response must be a dictionary."}

    if response.get("status") != "ok":
        error = response.get("error", "AI response status is not ok.")
        return {"status": "error", "error": error}

    raw = response.get("response")

    if isinstance(raw, dict):
        return {"status": "ok", "data": raw}

    if isinstance(raw, str):
        try:
            return {"status": "ok", "data": json.loads(raw)}
        except json.JSONDecodeError:
            return {"status": "error", "error": "AI response is not valid JSON."}

    return {"status": "error", "error": "AI response field must be JSON string or dictionary."}


def _has_filled_section(sections: dict[str, Any]) -> bool:
    for value in sections.values():
        if isinstance(value, str) and value.strip():
            return True

        if isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value):
            return True

    return False


def _validate_brief(brief: Any) -> Optional[str]:
    if not isinstance(brief, dict):
        return "Brief must be a dictionary."

    topic = brief.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        return "topic is required."

    lesson_description = brief.get("lesson_description")
    if not isinstance(lesson_description, str) or not lesson_description.strip():
        return "lesson_description is required."

    sections = brief.get("sections")
    if not isinstance(sections, dict) or not _has_filled_section(sections):
        return "At least one section is required."

    return None


def _clean_pairs(pairs: Any) -> list[dict[str, str]]:
    if not isinstance(pairs, list):
        return []

    clean = []
    used_left = set()
    used_right = set()

    for pair in pairs:
        if not isinstance(pair, dict):
            continue

        left = pair.get("word", pair.get("left"))
        right = pair.get("translation", pair.get("right"))

        if not isinstance(left, str) or not isinstance(right, str):
            continue

        left = left.strip()
        right = right.strip()

        if not left or not right:
            continue

        left_key = left.lower()
        right_key = right.lower()

        if left_key in used_left or right_key in used_right:
            continue

        used_left.add(left_key)
        used_right.add(right_key)

        if "word" in pair or "translation" in pair:
            clean.append({"word": left, "translation": right})
        else:
            clean.append({"left": left, "right": right})

    return clean


def _format_fill_gaps(text: str, answers: list[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None

    if text.count("___") != len(answers):
        return None

    formatted_text = text

    for answer in answers:
        if not isinstance(answer, str) or not answer.strip():
            return None

        formatted_text = formatted_text.replace("___", "{{" + answer.strip() + "}}", 1)

    return formatted_text


def _validate_test_questions(questions: Any, strict: bool) -> Optional[list[dict[str, Any]]]:
    if not isinstance(questions, list):
        return None

    clean = []

    for question in questions:
        if not isinstance(question, dict):
            continue

        options = question.get("options")
        if not isinstance(options, list):
            continue

        has_correct = any(
            isinstance(option, dict) and option.get("is_correct") is True
            for option in options
        )

        if strict and not has_correct:
            return None

        if has_correct:
            clean.append(question)

    return clean


def _clean_true_false(statements: Any) -> list[dict[str, Any]]:
    if not isinstance(statements, list):
        return []

    return [
        statement
        for statement in statements
        if isinstance(statement, dict)
           and isinstance(statement.get("statement"), str)
           and isinstance(statement.get("is_true"), bool)
    ]


def _build_vocabulary_prompt(
        topic: str,
        vocabulary: list[str],
        previous_error: Optional[str] = None,
) -> str:
    payload = {
        "lesson_topic": topic,
        "task": "Form a vocabulary section. It should include a word list, then 1 or 2 tasks to practice.",
        "vocabulary": ", ".join(vocabulary),
        "available_task_types": {
            "word_list": {
                "json": {"type": "word_list", "pairs": [{"word": "string", "translation": "string"}]},
                "rules": [
                    "Translation should be into Russian.",
                    "Give one exact translation, not options.",
                ],
            },
            "match_cards": {
                "json": {"type": "match_cards", "pairs": [{"left": "string", "right": "string"}]},
                "rules": [
                    "You may use it as matching words and phrases or matching card with its meaning.",
                ],
            },
            "fill_gaps": {
                "json": {"type": "fill_gaps", "text": "string", "answers": ["string"]},
                "rules": [
                    "Give from 6 to 10 gaps.",
                    "Mark gaps as ___.",
                    "Unless explicitly requested otherwise, use independent sentences instead of one connected text.",
                    "Each sentence must be on a separate line.",
                    "Prefer exactly one gap per sentence.",
                    "You mustn't use more than one gap per sentence.",
                    "You must use line breaking (\\n).",
                ],
            },
        },
        "response_schema": {"tasks": [{"type": "string"}]},
        "rules": ["Return only valid JSON.", "Do not add markdown outside JSON."],
    }

    if previous_error:
        payload["previous_error"] = previous_error

    return _dump_prompt(payload)


def _build_grammar_prompt(
        topic: str,
        grammar: str,
        previous_error: Optional[str] = None,
) -> str:
    payload = {
        "lesson_topic": topic,
        "grammar_topic": grammar,
        "task": (
            "You give materials on the grammar topic. Think like a methodist: probably, "
            "you need to divide tasks into blocks. For example, in the first block "
            "give examples and a task for the first part of the topic, in the second "
            "block - for the second part of the topic."
        ),
        "available_task_types": {
            "note": {
                "json": {"type": "note", "content": "string"},
                "rules": [
                    "content must include markdown.",
                    "Bold and italic are available.",
                    "You must use line breaking (\\n).",
                    "Do not give explanations or tasks in note.",
                    "Just give some examples.",
                ],
            },
            "fill_gaps": {
                "json": {"type": "fill_gaps", "text": "string", "answers": ["string"]},
                "rules": [
                    "Mark gaps as ___.",
                    "Give base words if it is needed for the task.",
                    "Give from 6 to 10 gaps.",
                    "You mustn't use more than one gap per sentence.",
                    "You must use line breaking (\\n).",
                ],
            },
            "test": {
                "json": {
                    "type": "test",
                    "questions": [
                        {
                            "question": "string",
                            "options": [{"option": "string", "is_correct": "boolean"}],
                        }
                    ],
                },
                "rules": [
                    "Questions must have similar format.",
                    "Questions must be short and precise.",
                    "Keep the question field free of extra instructions.",
                    "Give 4-6 questions.",
                ],
            },
            "true_false": {
                "json": {
                    "type": "true_false",
                    "statements": [{"statement": "string", "is_true": "boolean"}],
                },
                "rules": [
                    "Statements must be short and clear.",
                    "Give 3-6 statements.",
                ],
            },
        },
        "response_schema": {"tasks": [{"type": "string"}]},
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
            "Task types may repeat in different blocks.",
        ],
    }

    if previous_error:
        payload["previous_error"] = previous_error

    return _dump_prompt(payload)


def _build_reading_prompt(
        brief: dict[str, Any],
        reading: str,
        previous_error: Optional[str] = None,
) -> str:
    sections = brief.get("sections", {})

    payload = {
        "lesson_topic": brief.get("topic", ""),
        "task": (
            "You are creating a reading section. First, generate an interesting article topic. "
            "Second, generate this article. Third, add a task to check student's understanding."
        ),
        "lesson_vocabulary": sections.get("vocabulary", []),
        "section_instruction": reading,
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
            "You don't need to use all the vocabulary or grammar.",
            "Just take some key points.",
        ],
        "available_task_types": {
            "reading_article": {
                "json": {"type": "reading_article", "content": "string"},
                "rules": ["You can use markdown bold a little.", "You must actively use \\n for line breaking.", "Don't make the article too short.", "Try to make article longer and use several paragraphs."],
            },
            "test": {
                "json": {
                    "type": "test",
                    "questions": [
                        {
                            "question": "string",
                            "options": [{"option": "string", "is_correct": "boolean"}],
                        }
                    ],
                },
                "rules": [
                    "Questions must have similar format.",
                    "Questions must be short and precise.",
                    "Give 4-6 questions.",
                ],
            },
            "true_false": {
                "json": {
                    "type": "true_false",
                    "statements": [{"statement": "string", "is_true": "boolean"}],
                },
                "rules": [
                    "Statements must be short and clear.",
                    "Give 3-6 statements.",
                ],
            },
        },
        "response_schema": {"tasks": [{"type": "string"}]},
    }

    if previous_error:
        payload["previous_error"] = previous_error

    return _dump_prompt(payload)


def _build_listening_prompt(
        brief: dict[str, Any],
        listening: str,
        previous_error: Optional[str] = None,
) -> str:
    sections = brief.get("sections", {})

    payload = {
        "lesson_topic": brief.get("topic", ""),
        "task": (
            "You are creating a listening section. First, generate an interesting audio topic. "
            "Second, generate a listening script. Third, add a task to check student's understanding."
        ),
        "lesson_vocabulary": sections.get("vocabulary", []),
        "section_instruction": listening,
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
            "Try to make script longer.",
            "You don't need to use all the vocabulary or grammar.",
            "Just take some key points.",
        ],
        "available_task_types": {
            "listening_script": {
                "json": {
                    "type": "listening_script",
                    "mode": "monologue | dialogue",
                    "script": [{"speaker": "string", "text": "string"}],
                },
            },
            "test": {
                "json": {
                    "type": "test",
                    "questions": [
                        {
                            "question": "string",
                            "options": [{"option": "string", "is_correct": "boolean"}],
                        }
                    ],
                },
                "rules": [
                    "Questions must have similar format.",
                    "Questions must be short and precise.",
                    "Give 4-6 questions.",
                ],
            },
            "true_false": {
                "json": {
                    "type": "true_false",
                    "statements": [{"statement": "string", "is_true": "boolean"}],
                },
                "rules": [
                    "Statements must be short and clear.",
                    "Give 3-6 statements.",
                ],
            },
        },
        "response_schema": {"tasks": [{"type": "string"}]},
    }

    if previous_error:
        payload["previous_error"] = previous_error

    return _dump_prompt(payload)


def _build_speaking_prompt(
        brief: dict[str, Any],
        speaking: str,
        previous_error: Optional[str] = None,
) -> str:
    sections = brief.get("sections", {})

    payload = {
        "lesson_topic": brief.get("topic", ""),
        "lesson_vocabulary": sections.get("vocabulary", []),
        "task": "You are creating a speaking section for a personal lesson.",
        "section_instruction": speaking,
        "available_task_types": {
            "image_description": {
                "json": {"type": "image_description", "image_description": "string"},
                "rules": ["Use this if an image would help the discussion."],
            },
            "speaking_questions": {
                "json": {"type": "speaking_questions", "speaking_questions": ["string"]},
                "rules": ["Return speaking_questions as an array of strings."],
            },
        },
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
            "You don't need to use all the vocabulary or grammar.",
            "Just take some key points.",
            "Your speaking questions should be short, clear and discussion-related.",
            "Return tasks as a flat array.",
            "Do not wrap tasks in option/name objects.",
        ],
        "response_schema": {
            "tasks": [
                {"type": "image_description", "image_description": "string"},
                {"type": "speaking_questions", "speaking_questions": ["string"]},
            ],
        },
    }

    if previous_error:
        payload["previous_error"] = previous_error

    return _dump_prompt(payload)


async def _call_ai_json(section_name: str, prompt_builder, *args) -> dict[str, Any]:
    previous_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = prompt_builder(*args, previous_error)
        response = await asyncio.to_thread(call_ai, prompt)
        parsed = _parse_ai_json(response)

        if parsed["status"] == "ok" and isinstance(parsed["data"].get("tasks"), list):
            logger.info(
                "Generated raw %s section on attempt %s with %s task(s)",
                section_name,
                attempt,
                len(parsed["data"]["tasks"]),
            )
            return parsed["data"]

        if parsed["status"] == "ok":
            previous_error = "AI JSON response does not contain a tasks list."
        else:
            previous_error = parsed.get("error", "Invalid tasks response.")

        logger.warning(
            "Failed to generate raw %s section on attempt %s/%s: %s",
            section_name,
            attempt,
            MAX_ATTEMPTS,
            previous_error,
        )

    logger.error("Failed to generate raw %s section after %s attempts", section_name, MAX_ATTEMPTS)
    return {"tasks": []}


def _payload_preview(payload: Any) -> str:
    try:
        preview = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        preview = repr(payload)

    if len(preview) > LOG_PAYLOAD_PREVIEW_LENGTH:
        return f"{preview[:LOG_PAYLOAD_PREVIEW_LENGTH]}..."

    return preview


def _log_skipped_task(section_name: str, task_type: Any, reason: str, payload: Any) -> None:
    logger.warning(
        "Skipped %s task in %s section: %s. Received: %s",
        task_type or "unknown",
        section_name,
        reason,
        _payload_preview(payload),
    )


def _log_section_result(section_name: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if tasks:
        logger.info("Generated %s section with %s accepted task(s)", section_name, len(tasks))
    else:
        logger.warning("Generated %s section has no accepted tasks", section_name)

    return {"tasks": tasks}


async def _process_fill_gaps_task(
        task: dict[str, Any],
        mode: str,
        section_name: str,
) -> Optional[dict[str, Any]]:
    answers = task.get("answers")

    if not isinstance(answers, list):
        _log_skipped_task(section_name, "fill_gaps", "answers must be a list", task)
        return None

    formatted_text = _format_fill_gaps(task.get("text", ""), answers)

    if not formatted_text:
        _log_skipped_task(
            section_name,
            "fill_gaps",
            "text must contain the same number of ___ gaps as non-empty answers",
            task,
        )
        return None

    return {
        "type": "fill_gaps",
        "text": formatted_text,
        "mode": mode,
        "answers": answers,
    }


async def _generate_vocabulary_section(topic: str, vocabulary: list[str]) -> dict[str, Any]:
    data = await _call_ai_json("vocabulary", _build_vocabulary_prompt, topic, vocabulary)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            _log_skipped_task("vocabulary", None, "task must be a dictionary", task)
            continue

        task_type = task.get("type")

        if task_type == "word_list":
            pairs = _clean_pairs(task.get("pairs"))
            if pairs:
                tasks.append({"type": "word_list", "pairs": pairs})
            else:
                _log_skipped_task("vocabulary", task_type, "pairs are empty or invalid", task)

        elif task_type == "match_cards":
            pairs = _clean_pairs(task.get("pairs"))
            if pairs:
                tasks.append({"type": "match_cards", "pairs": pairs})
            else:
                _log_skipped_task("vocabulary", task_type, "pairs are empty or invalid", task)

        elif task_type == "fill_gaps":
            fill_gaps = await _process_fill_gaps_task(task, "open", "vocabulary")
            if fill_gaps:
                tasks.append(fill_gaps)
        else:
            _log_skipped_task("vocabulary", task_type, "unsupported task type", task)

    return _log_section_result("vocabulary", tasks)


async def _generate_grammar_section(topic: str, grammar: str) -> dict[str, Any]:
    data = await _call_ai_json("grammar", _build_grammar_prompt, topic, grammar)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            _log_skipped_task("grammar", None, "task must be a dictionary", task)
            continue

        task_type = task.get("type")

        if task_type == "note":
            content = task.get("content")
            if isinstance(content, str) and content.strip():
                tasks.append({"type": "note", "content": content})
            else:
                _log_skipped_task("grammar", task_type, "content is empty or invalid", task)

        elif task_type == "fill_gaps":
            fill_gaps = await _process_fill_gaps_task(task, "closed", "grammar")
            if fill_gaps:
                tasks.append(fill_gaps)

        elif task_type == "test":
            questions = _validate_test_questions(task.get("questions"), strict=False)
            if questions:
                tasks.append({"type": "test", "questions": questions})
            else:
                _log_skipped_task("grammar", task_type, "questions are empty or invalid", task)

        elif task_type == "true_false":
            statements = _clean_true_false(task.get("statements"))
            if statements:
                tasks.append({"type": "true_false", "statements": statements})
            else:
                _log_skipped_task("grammar", task_type, "statements are empty or invalid", task)
        else:
            _log_skipped_task("grammar", task_type, "unsupported task type", task)

    return _log_section_result("grammar", tasks)


async def _generate_reading_section(brief: dict[str, Any], reading: str) -> dict[str, Any]:
    data = await _call_ai_json("reading", _build_reading_prompt, brief, reading)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            _log_skipped_task("reading", None, "task must be a dictionary", task)
            continue

        task_type = task.get("type")

        if task_type == "reading_article":
            content = task.get("content")
            if isinstance(content, str) and content.strip():
                tasks.append({"type": "note", "content": content})
            else:
                _log_skipped_task("reading", task_type, "content is empty or invalid", task)

        elif task_type == "test":
            questions = _validate_test_questions(task.get("questions"), strict=False)
            if questions:
                tasks.append({"type": "test", "questions": questions})
            else:
                _log_skipped_task("reading", task_type, "questions are empty or invalid", task)

        elif task_type == "true_false":
            statements = _clean_true_false(task.get("statements"))
            if statements:
                tasks.append({"type": "true_false", "statements": statements})
            else:
                _log_skipped_task("reading", task_type, "statements are empty or invalid", task)
        else:
            _log_skipped_task("reading", task_type, "unsupported task type", task)

    return _log_section_result("reading", tasks)


def _script_to_text(script: Any) -> str:
    if not isinstance(script, list):
        return ""

    lines = []

    for item in script:
        if not isinstance(item, dict):
            continue

        speaker = item.get("speaker", "")
        text = item.get("text", "")

        if isinstance(text, str) and text.strip():
            if isinstance(speaker, str) and speaker.strip():
                lines.append(f"**{speaker.strip()}:** {text.strip()}")
            else:
                lines.append(text.strip())

    return "\n".join(lines)


async def _generate_audio_file(mode: str, script: list[dict[str, str]]) -> dict[str, Any]:
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(generate_audio, mode, script),
            timeout=MEDIA_TIMEOUT,
        )
    except TimeoutError:
        response = {"status": "error", "error": f"Audio generation timed out after {MEDIA_TIMEOUT} seconds."}

    if isinstance(response, dict) and response.get("status") == "ok" and response.get("audio_base64"):
        logger.info("Generated listening audio file")
        return {
            "type": "file",
            "file_type": "audio",
            "base64": response["audio_base64"],
        }

    error = response.get("error", "Audio generation returned an invalid response") if isinstance(response, dict) else "Audio generation returned a non-dictionary response"
    logger.warning("Audio file generation failed, falling back to transcript note: %s", error)
    transcript = _script_to_text(script)

    return {
        "type": "note",
        "content": (
            "**К сожалению, генерация аудио временно недоступна.**\n\n"
            "Вы можете воспользоваться внешними сервисами для генерации. "
            "Например, https://notegpt.io/ai-podcast-generator\n\n"
            f"Транскрипт записи:\n{transcript}"
        ),
    }


async def _generate_listening_section(brief: dict[str, Any], listening: str) -> dict[str, Any]:
    data = await _call_ai_json("listening", _build_listening_prompt, brief, listening)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            _log_skipped_task("listening", None, "task must be a dictionary", task)
            continue

        task_type = task.get("type")

        if task_type == "listening_script":
            mode = task.get("mode")
            script = task.get("script")

            if mode in {"monologue", "dialogue"} and isinstance(script, list):
                tasks.append(await _generate_audio_file(mode, script))
            else:
                _log_skipped_task("listening", task_type, "mode must be monologue/dialogue and script must be a list", task)

        elif task_type == "test":
            questions = _validate_test_questions(task.get("questions"), strict=False)
            if questions:
                tasks.append({"type": "test", "questions": questions})
            else:
                _log_skipped_task("listening", task_type, "questions are empty or invalid", task)

        elif task_type == "true_false":
            statements = _clean_true_false(task.get("statements"))
            if statements:
                tasks.append({"type": "true_false", "statements": statements})
            else:
                _log_skipped_task("listening", task_type, "statements are empty or invalid", task)
        else:
            _log_skipped_task("listening", task_type, "unsupported task type", task)

    return _log_section_result("listening", tasks)


async def _generate_image_file(description: str) -> dict[str, Any]:
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(generate_image, description),
            timeout=MEDIA_TIMEOUT,
        )
    except TimeoutError:
        response = {"status": "error", "error": f"Image generation timed out after {MEDIA_TIMEOUT} seconds."}

    if isinstance(response, dict) and response.get("status") == "ok" and response.get("image_base64"):
        logger.info("Generated speaking image file")
        return {
            "type": "file",
            "file_type": "image",
            "base64": response["image_base64"],
        }

    error = response.get("error", "Image generation returned an invalid response") if isinstance(response, dict) else "Image generation returned a non-dictionary response"
    logger.warning("Image file generation failed, falling back to description note: %s", error)
    return {
        "type": "note",
        "content": (
            "**К сожалению, генерация изображений временно недоступна.**\n\n"
            "Вы можете воспользоваться внешними сервисами для генерации. "
            "Например, https://notegpt.io/ai-image-generator\n\n"
            f"**Описание изображения:** {description}"
        ),
    }


def _build_speaking_note(questions: list[str]) -> dict[str, str]:
    lines = [f"**Discuss the questions**", ""]

    for index, question in enumerate(questions, start=1):
        if isinstance(question, str) and question.strip():
            lines.append(f"{index}. {question.strip()}")

    return {
        "type": "note",
        "content": "\n".join(lines),
    }


async def _generate_speaking_section(brief: dict[str, Any], speaking: str) -> dict[str, Any]:
    data = await _call_ai_json("speaking", _build_speaking_prompt, brief, speaking)

    image_description = None
    questions = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            _log_skipped_task("speaking", None, "task must be a dictionary", task)
            continue

        if task.get("type") == "image_description":
            description = task.get("image_description")
            if isinstance(description, str) and description.strip():
                image_description = description.strip()
            else:
                _log_skipped_task("speaking", "image_description", "image_description is empty or invalid", task)

        elif task.get("type") == "speaking_questions":
            raw_questions = task.get("speaking_questions")
            if isinstance(raw_questions, list):
                questions = [
                    question.strip()
                    for question in raw_questions
                    if isinstance(question, str) and question.strip()
                ]
                if not questions:
                    _log_skipped_task("speaking", "speaking_questions", "speaking_questions contains no valid questions", task)
            else:
                _log_skipped_task("speaking", "speaking_questions", "speaking_questions must be a list", task)
        else:
            _log_skipped_task("speaking", task.get("type"), "unsupported task type", task)

    tasks = []

    if image_description:
        tasks.append(await _generate_image_file(image_description))

    if questions:
        tasks.append(_build_speaking_note(questions))

    return _log_section_result("speaking", tasks)


async def _generate_sections(brief: dict[str, Any]) -> dict[str, Any]:
    topic = brief["topic"]
    sections = brief.get("sections", {})
    jobs = {}

    vocabulary = sections.get("vocabulary")
    valid_vocabulary = [
        item
        for item in vocabulary
        if isinstance(item, str) and item.strip()
    ] if isinstance(vocabulary, list) else []

    if len(valid_vocabulary) >= 4:
        jobs["vocabulary"] = _generate_vocabulary_section(topic, vocabulary)
    elif valid_vocabulary:
        logger.warning(
            "Vocabulary section was requested but not generated: at least 4 vocabulary items are required, got %s",
            len(valid_vocabulary),
        )

    grammar = sections.get("grammar")
    if isinstance(grammar, str) and grammar.strip():
        jobs["grammar"] = _generate_grammar_section(topic, grammar)

    reading = sections.get("reading")
    if isinstance(reading, str) and reading.strip():
        jobs["reading"] = _generate_reading_section(brief, reading)

    listening = sections.get("listening")
    if isinstance(listening, str) and listening.strip():
        jobs["listening"] = _generate_listening_section(brief, listening)

    speaking = sections.get("speaking")
    if isinstance(speaking, str) and speaking.strip():
        jobs["speaking"] = _generate_speaking_section(brief, speaking)

    if not jobs:
        logger.warning("No task sections were scheduled for generation")
        return {}

    results = await asyncio.gather(*jobs.values(), return_exceptions=True)
    sections_result = {}

    for section_name, result in zip(jobs.keys(), results):
        if isinstance(result, Exception):
            logger.error(
                "Failed to generate %s section",
                section_name,
                exc_info=(type(result), result, result.__traceback__),
            )
            sections_result[section_name] = {"tasks": []}
        else:
            sections_result[section_name] = result

    return sections_result


def generate_tasks(brief: dict[str, Any]) -> dict[str, Any]:
    validation_error = _validate_brief(brief)

    if validation_error:
        logger.error("Task generation rejected invalid brief: %s", validation_error)
        return {
            "status": "error",
            "error": validation_error,
        }

    logger.info("Starting task generation for topic: %s", brief["topic"])
    sections = asyncio.run(_generate_sections(brief))
    empty_sections = [
        name
        for name, section in sections.items()
        if not isinstance(section.get("tasks"), list) or not section["tasks"]
    ]

    if empty_sections:
        logger.warning("Task generation finished with empty section(s): %s", ", ".join(empty_sections))

    return {
        "status": "ok",
        "topic": brief["topic"],
        "sections": sections,
    }

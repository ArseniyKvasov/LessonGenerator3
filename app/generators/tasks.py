import asyncio
import json
from typing import Any, Optional

from app.clients.groq import call_ai
from app.clients.pollinations import generate_audio, generate_image

MAX_ATTEMPTS = 2
MEDIA_TIMEOUT = 60


def _dump_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_ai_json(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"status": "error", "error": "AI response must be a dictionary."}

    if response.get("status") != "ok":
        return {"status": "error", "error": "AI response status is not ok."}

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
                    "Give from 5 to 10 gaps.",
                    "Mark gaps as ___.",
                    "Use one gap per sentence.",
                    "Use \\n for line breaking.",
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
                    "Use \\n for line breaking.",
                    "Do not give explanations or tasks in note.",
                    "Just give some examples.",
                ],
            },
            "fill_gaps": {
                "json": {"type": "fill_gaps", "text": "string", "answers": ["string"]},
                "rules": [
                    "Mark gaps as ___.",
                    "Give base words if it is needed for the task.",
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
        "lesson_grammar": sections.get("grammar", ""),
        "section_instruction": reading,
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
            "You don't need to use all the vocabulary or grammar.",
            "Just take some key points.",
        ],
        "available_task_types": {
            "reading_text": {
                "json": {"type": "reading_text", "content": "string"},
                "rules": ["Use markdown.", "Bold is available."],
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
        "lesson_grammar": sections.get("grammar", ""),
        "section_instruction": listening,
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
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
        "lesson_grammar": sections.get("grammar", ""),
        "task": "You are creating a speaking section for a personal lesson.",
        "section_instruction": speaking,
        "options": [
            {
                "name": "discussion_by_image",
                "tasks": [
                    {"type": "image_description", "image_description": "string"},
                    {"type": "speaking_questions", "speaking_questions": ["string"]},
                ],
            },
            {
                "name": "speaking_cards",
                "tasks": [
                    {"type": "speaking_questions", "speaking_questions": ["string"]},
                ],
            },
        ],
        "rules": [
            "Return only valid JSON.",
            "Do not add markdown outside JSON.",
            "You don't need to use all the vocabulary or grammar.",
            "Just take some key points.",
            "Your speaking questions should be short, clear and discussion-related.",
        ],
        "response_schema": {"tasks": [{"type": "string"}]},
    }

    if previous_error:
        payload["previous_error"] = previous_error

    return _dump_prompt(payload)


async def _call_ai_json(prompt_builder, *args) -> dict[str, Any]:
    previous_error = None

    for _ in range(MAX_ATTEMPTS):
        prompt = prompt_builder(*args, previous_error)
        response = await asyncio.to_thread(call_ai, prompt)
        parsed = _parse_ai_json(response)

        if parsed["status"] == "ok" and isinstance(parsed["data"].get("tasks"), list):
            return parsed["data"]

        previous_error = parsed.get("error", "Invalid tasks response.")

    return {"tasks": []}


async def _process_fill_gaps_task(task: dict[str, Any], mode: str) -> Optional[dict[str, Any]]:
    answers = task.get("answers")

    if not isinstance(answers, list):
        return None

    formatted_text = _format_fill_gaps(task.get("text", ""), answers)

    if not formatted_text:
        return None

    return {
        "type": "fill_gaps",
        "text": formatted_text,
        "mode": mode,
        "answers": answers,
    }


async def _generate_vocabulary_section(topic: str, vocabulary: list[str]) -> dict[str, Any]:
    data = await _call_ai_json(_build_vocabulary_prompt, topic, vocabulary)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue

        task_type = task.get("type")

        if task_type == "word_list":
            pairs = _clean_pairs(task.get("pairs"))
            if pairs:
                tasks.append({"type": "word_list", "pairs": pairs})

        elif task_type == "match_cards":
            pairs = _clean_pairs(task.get("pairs"))
            if pairs:
                tasks.append({"type": "match_cards", "pairs": pairs})

        elif task_type == "fill_gaps":
            fill_gaps = await _process_fill_gaps_task(task, "open")
            if fill_gaps:
                tasks.append(fill_gaps)

    return {"tasks": tasks}


async def _generate_grammar_section(topic: str, grammar: str) -> dict[str, Any]:
    data = await _call_ai_json(_build_grammar_prompt, topic, grammar)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue

        task_type = task.get("type")

        if task_type == "note":
            content = task.get("content")
            if isinstance(content, str) and content.strip():
                tasks.append({"type": "note", "content": content})

        elif task_type == "fill_gaps":
            fill_gaps = await _process_fill_gaps_task(task, "closed")
            if fill_gaps:
                tasks.append(fill_gaps)

        elif task_type == "test":
            questions = _validate_test_questions(task.get("questions"), strict=False)
            if questions:
                tasks.append({"type": "test", "questions": questions})

        elif task_type == "true_false":
            statements = _clean_true_false(task.get("statements"))
            if statements:
                tasks.append({"type": "true_false", "statements": statements})

    return {"tasks": tasks}


async def _generate_reading_section(brief: dict[str, Any], reading: str) -> dict[str, Any]:
    data = await _call_ai_json(_build_reading_prompt, brief, reading)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue

        task_type = task.get("type")

        if task_type == "reading_text":
            content = task.get("content")
            if isinstance(content, str) and content.strip():
                tasks.append({"type": "note", "content": content})

        elif task_type == "test":
            questions = _validate_test_questions(task.get("questions"), strict=False)
            if questions:
                tasks.append({"type": "test", "questions": questions})

        elif task_type == "true_false":
            statements = _clean_true_false(task.get("statements"))
            if statements:
                tasks.append({"type": "true_false", "statements": statements})

    return {"tasks": tasks}


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
                lines.append(f"{speaker.strip()}: {text.strip()}")
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
        response = {"status": "error"}

    if isinstance(response, dict) and response.get("status") == "ok" and response.get("audio_base64"):
        return {
            "type": "file",
            "file_type": "audio",
            "base64": response["audio_base64"],
        }

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
    data = await _call_ai_json(_build_listening_prompt, brief, listening)
    tasks = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue

        task_type = task.get("type")

        if task_type == "listening_script":
            mode = task.get("mode")
            script = task.get("script")

            if mode in {"monologue", "dialogue"} and isinstance(script, list):
                tasks.append(await _generate_audio_file(mode, script))

        elif task_type == "test":
            questions = _validate_test_questions(task.get("questions"), strict=False)
            if questions:
                tasks.append({"type": "test", "questions": questions})

        elif task_type == "true_false":
            statements = _clean_true_false(task.get("statements"))
            if statements:
                tasks.append({"type": "true_false", "statements": statements})

    return {"tasks": tasks}


async def _generate_image_file(description: str) -> dict[str, Any]:
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(generate_image, description),
            timeout=MEDIA_TIMEOUT,
        )
    except TimeoutError:
        response = {"status": "error"}

    if isinstance(response, dict) and response.get("status") == "ok" and response.get("image_base64"):
        return {
            "type": "file",
            "file_type": "image",
            "base64": response["image_base64"],
        }

    return {
        "type": "note",
        "content": (
            "**К сожалению, генерация изображений временно недоступна.**\n\n"
            "Вы можете воспользоваться внешними сервисами для генерации. "
            "Например, https://notegpt.io/ai-image-generator\n\n"
            f"**Описание изображения:** {description}"
        ),
    }


def _build_speaking_note(questions: list[str], has_image: bool) -> dict[str, str]:
    instruction = "Discuss the image" if has_image else "Discuss the questions"

    lines = [f"**{instruction}**", ""]

    for index, question in enumerate(questions, start=1):
        if isinstance(question, str) and question.strip():
            lines.append(f"{index}. {question.strip()}")

    return {
        "type": "note",
        "content": "\n".join(lines),
    }


async def _generate_speaking_section(brief: dict[str, Any], speaking: str) -> dict[str, Any]:
    data = await _call_ai_json(_build_speaking_prompt, brief, speaking)

    image_description = None
    questions = []

    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue

        if task.get("type") == "image_description":
            description = task.get("image_description")
            if isinstance(description, str) and description.strip():
                image_description = description.strip()

        elif task.get("type") == "speaking_questions":
            raw_questions = task.get("speaking_questions")
            if isinstance(raw_questions, list):
                questions = [
                    question.strip()
                    for question in raw_questions
                    if isinstance(question, str) and question.strip()
                ]

    tasks = []

    if image_description:
        tasks.append(await _generate_image_file(image_description))

    if questions:
        tasks.append(_build_speaking_note(questions, bool(image_description)))

    return {"tasks": tasks}


async def _generate_sections(brief: dict[str, Any]) -> dict[str, Any]:
    topic = brief["topic"]
    sections = brief.get("sections", {})
    jobs = {}

    vocabulary = sections.get("vocabulary")
    if isinstance(vocabulary, list) and len(
            [item for item in vocabulary if isinstance(item, str) and item.strip()]) >= 4:
        jobs["vocabulary"] = _generate_vocabulary_section(topic, vocabulary)

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

    results = await asyncio.gather(*jobs.values())

    return dict(zip(jobs.keys(), results))


def generate_tasks(brief: dict[str, Any]) -> dict[str, Any]:
    validation_error = _validate_brief(brief)

    if validation_error:
        return {
            "status": "error",
            "error": validation_error,
        }

    sections = asyncio.run(_generate_sections(brief))

    return {
        "status": "ok",
        "topic": brief["topic"],
        "sections": sections,
    }

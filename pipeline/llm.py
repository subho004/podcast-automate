import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from typing import List
import re
import logging

# Load environment from .env if present
load_dotenv()
logger = logging.getLogger(__name__)
import json

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
else:
    logger.debug("OPENAI_API_KEY not found in environment after loading .env")

# instantiate modern OpenAI client
_openai_client = OpenAI()


def complete_system(prompt: str, model: str = "gpt-5-mini", temperature: float = 1) -> str:
    if not (OPENAI_KEY or os.getenv("OPENAI_API_KEY") or openai.api_key):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    # Use the new OpenAI client API
    resp = _openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce concise structured JSON responses when asked."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    # New API returns choices with message.content
    return resp.choices[0].message.content.strip()


def generate_outline(query: str, scenes: int | None = 3) -> str:
    """Generate an episode outline.

    If `scenes` is None, ask the model to choose an appropriate scene count
    (recommend between 3 and 6) and include the chosen number in the JSON.
    Returns the raw model string (preferably JSON).
    """
    if scenes is None:
        prompt = (
            f"Create an episode title and choose an appropriate number of short scenes (recommended 3-6) "
            f"for the user query: \"{query}\". Return JSON with keys: title, scenes_count, scenes "
            "(list of {id, title, blurb})."
        )
    else:
        prompt = (
            f"Create an episode title and {scenes} short scenes for the user query: \"{query}\". "
            "Return in JSON with keys: title, scenes (list of {id, title, blurb})."
        )
    return complete_system(prompt)


SUPPORTED_EMOTION_TAGS = [
    "<laugh>", "<laugh_harder>", "<sigh>", "<chuckle>", "<gasp>", "<angry>",
    "<excited>", "<whisper>", "<cry>", "<scream>", "<sing>", "<snort>",
    "<exhale>", "<gulp>", "<giggle>", "<sarcastic>", "<curious>",
]

# pattern to capture tags like <laugh> or <laugh_harder>
EMOTION_TAG_PATTERN = re.compile(r"<[^>]+>")


def validate_maya_text(text: str):
    """Validate Maya-formatted `text` for allowed emotion tags.

    Returns (is_valid, sanitized_text, unknown_tags_set)
    """
    if not text:
        return True, text, set()
    tags = set(EMOTION_TAG_PATTERN.findall(text))
    allowed = set(SUPPORTED_EMOTION_TAGS)
    unknown = tags - allowed
    if not unknown:
        return True, text, set()
    # sanitize by removing unknown tags
    sanitized = text
    for t in unknown:
        sanitized = sanitized.replace(t, "")
    return False, sanitized, unknown


def scene_to_maya(scene: dict, voice_hint: str | None = None, model: str = "gpt-4o-mini") -> dict:
    """Convert a scene (id/title/blurb) into a Maya1 formatted `text` and a short `voice_description`.

    Returns a dict: {"text": str, "voice_description": str, "raw": str}
    The `text` is the speech content to pass to the Maya1 prompt builder, and
    `voice_description` is a short natural-language description of the voice/style.
    """
    blurb = scene.get("blurb") or scene.get("text") or scene.get("title", "")
    title = scene.get("title", "")

    # Maya1 expects a `voice description` (natural-language) and inline emotion tags inside the text.
    # We'll ask the LLM to return JSON with 2 keys:
    # - text: the spoken content for Maya1. Use inline emotion tags like <laugh>, <sigh>, <whisper>, <cry>, etc.
    # - voice_description: short natural-language voice brief (e.g. "Male, 30s, warm, conversational").
    # format supported emotions for prompt
    supported_emotions = ", ".join(SUPPORTED_EMOTION_TAGS)

    prompt = (
        "You are an assistant that converts a short scene blurb into Maya1-ready speech text and voice metadata.\n"
        "Return strictly valid JSON (no additional commentary) with keys: text and voice_description.\n"
        "Requirements:\n"
        "- `text`: 1-3 short paragraphs (prefer concise spoken sentences). You may include inline emotion tags from the list below (place tags inline where the actor would express emotion).\n"
        "- `voice_description`: a 4-10 word natural-language brief describing voice (age, gender, tone, pacing).\n"
        "- Provide a brief `voice_description` only; do not provide a separate maya_description.\n"
        "- Use emotion tags generously where they help convey performance; prefer richer expressive tags in the `text`.\n"
        "- Return only the two JSON keys `text` and `voice_description` (do not add any other keys).\n"
        f"Supported emotion tags: {supported_emotions}\n\n"
        f"Scene title: {title}\n"
        f"Scene blurb: {blurb}\n"
    )

    if voice_hint:
        prompt += f"Prefer this voice hint: {voice_hint}\n"

    # Provide a short example to guide formatting
    prompt += (
        "\nExample output JSON:\n{\n  \"text\": \"Wow. This place looks even better than I imagined. <laugh>\",\n  \"voice_description\": \"Female, 30s, American, energetic host\"\n}\n"
    )

    raw = complete_system(prompt, model=model)

    def strip_code_fence(s: str) -> str:
        if not s:
            return s
        s = s.strip()
        # Remove fenced code blocks like ```json\n...\n``` or ```...
        if s.startswith("```"):
            # drop the leading fence and any language tag line
            parts = s.split("\n", 1)
            if len(parts) == 2:
                s = parts[1]
        # drop trailing fences
        if s.endswith("```"):
            s = s[:-3]
        # remove any remaining triple backticks or markdown markers
        s = s.replace("```json", "").replace("```", "")
        # strip surrounding quotes
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]
        return s.strip()

    def extract_json(s: str):
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            # try to locate a JSON object inside the text
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
        return None

    # clean raw response from code fences and try to extract JSON
    cleaned = strip_code_fence(raw)
    parsed = extract_json(cleaned) or extract_json(raw)

    if parsed:
        text_field = parsed.get("text") or parsed.get("prompt") or ""
        if isinstance(text_field, str):
            text = strip_code_fence(text_field)
        else:
            text = str(text_field)
        voice_description = parsed.get("voice_description") or parsed.get("voice") or ""
        voice_description = strip_code_fence(voice_description) if isinstance(voice_description, str) else str(voice_description)
    else:
        # fallback: treat the whole cleaned output as the maya text
        text = strip_code_fence(cleaned or raw)
        voice_description = voice_hint or "Neutral conversational voice, warm timbre."

    return {"text": text, "voice_description": voice_description, "raw": raw}


"""Utility functions for LLM operations."""

import json
import re
from typing import Any, List, Optional

try:  # pragma: no cover
    import dirtyjson
except ImportError:
    dirtyjson = None


def extract_llm_content(response: Any) -> str:
    """Extract content from LLM response.

    Handles different response formats:
    - Ollama: Returns string directly
    - Other providers: Returns message objects with .content attribute

    Args:
        response: LLM response (string or message object).

    Returns:
        Content string.
    """
    if isinstance(response, str):
        return response
    elif hasattr(response, "content"):
        return response.content
    elif hasattr(response, "text"):
        return response.text
    else:
        return str(response)


def _wrap_repeated_objects_in_list(content: str, key: str) -> str:
    """Wrap repeated dictionaries for a key into a JSON list."""

    pattern = rf'("{key}"\s*:\s*)(\{{[^{{}}]*\}}(?:\s*,\s*\{{[^{{}}]*\}})+)'

    def repl(match: re.Match) -> str:
        block = match.group(2)
        objects = re.findall(r"\{[^{}]*\}", block)
        return f'{match.group(1)}[{", ".join(objects)}]'

    return re.sub(pattern, repl, content, flags=re.DOTALL)


def parse_json_response(content: str) -> Any:
    """Parse JSON from LLM response, handling various formats.

    Args:
        content: Response content string that may contain JSON.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If JSON cannot be parsed.
    """
    # Try direct JSON parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON array or object in the content
    json_match = re.search(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    if dirtyjson is not None:
        try:
            return dirtyjson.loads(content)
        except Exception:
            pass

    # If all else fails, try to fix common JSON issues
    content_fixed = _wrap_repeated_objects_in_list(content, "evidence")

    # Remove trailing commas
    content_cleaned = re.sub(r",\s*}", "}", content_fixed)
    content_cleaned = re.sub(r",\s*]", "]", content_cleaned)

    try:
        return json.loads(content_cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from response: {e}. Content: {content[:200]}")


def normalize_markdown(content: str) -> str:
    """Normalize markdown text produced by LLMs.

    Args:
        content: Raw markdown string from an LLM output.

    Returns:
        Sanitized markdown string with consistent bullets and spacing.
    """
    if not content:
        return ""

    text = content.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", "    ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    bullet_pattern = re.compile(
        r"^\s*(?:[-*]|(?:\d+\.))\s+(?P<body>.+)$", re.MULTILINE
    )

    def _bullet_replacement(match: re.Match) -> str:
        body = match.group("body").strip()
        return f"- {body}"

    text = bullet_pattern.sub(_bullet_replacement, text)
    return text.strip()


def extract_list_items(content: str, max_items: Optional[int] = None) -> List[str]:
    """Extract bullet or ordered list items from markdown text.

    Args:
        content: Raw markdown string that may contain lists.
        max_items: Optional cap on number of extracted items.

    Returns:
        List of cleaned list entries.
    """
    normalized = normalize_markdown(content)
    if not normalized:
        return []

    items: List[str] = []
    current: List[str] = []

    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                items.append(" ".join(current).strip())
                current = []
            current.append(stripped[2:].strip())
        elif current and stripped:
            current.append(stripped)
        elif current and not stripped:
            items.append(" ".join(current).strip())
            current = []

        if max_items and len(items) >= max_items:
            break

    if current and (not max_items or len(items) < max_items):
        items.append(" ".join(current).strip())

    if not items and normalized:
        items = [normalized]

    if max_items:
        return items[:max_items]
    return items

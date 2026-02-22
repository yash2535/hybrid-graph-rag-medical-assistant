import requests

MAX_PROMPT_CHARS = 8000  # increase limit, phi3:mini can handle this

def call_ollama(
    prompt: str,
    model: str = "phi3:mini",
    temperature: float = 0.7,
) -> str:
    url = "http://localhost:11434/api/chat"

    # Smart truncation: keep structure, trim only the papers section
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = _smart_truncate(prompt, MAX_PROMPT_CHARS)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical AI assistant. "
                    "Answer clearly, concisely, and safely. "
                    "If evidence is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": temperature,
        },
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=460)

        if response.status_code != 200:
            return f"Ollama error {response.status_code}: {response.text}"

        data = response.json()
        return data["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running."

    except Exception as e:
        return f"Error calling Ollama: {e}"


def _smart_truncate(prompt: str, max_chars: int) -> str:
    """
    Truncate only the LITERATURE section to preserve
    patient, wearables, and medication data.
    """
    literature_marker = "RELEVANT MEDICAL LITERATURE"
    guidelines_marker = "USER QUESTION"

    if literature_marker in prompt and guidelines_marker in prompt:
        before_lit = prompt[:prompt.index(literature_marker)]
        from_question = prompt[prompt.index(guidelines_marker):]
        # Trim literature, keep everything else intact
        return before_lit + f"========================\n{literature_marker}\n========================\n[Truncated to fit context window]\n\n" + from_question

    # Fallback: dumb truncation only if structure not found
    return prompt[:max_chars]
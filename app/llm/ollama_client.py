import requests


def call_ollama(
    prompt: str,
    model: str = "phi3:mini",
    temperature: float = 0.7,
) -> str:
    url = "http://localhost:11434/api/chat"

    prompt = prompt[:3500]  # CRITICAL: prevent Ollama crash

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
        response = requests.post(url, json=payload, timeout=180)

        if response.status_code != 200:
            # IMPORTANT: show real Ollama error
            return f"Ollama error {response.status_code}: {response.text}"

        data = response.json()
        return data["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running."

    except Exception as e:
        return f"Error calling Ollama: {e}"

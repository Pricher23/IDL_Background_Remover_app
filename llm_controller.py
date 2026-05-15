import requests
import json
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"  # or llama3

def ask_llm(user_input, bg_names: list[str]):
    bg_list = "\n".join(f"- {n}" for n in bg_names)
    prompt = f"""
You are controlling a virtual background app.
Reply with a JSON object only — no explanation.

Available background images:
{bg_list}

Actions:
- {{"action": "set_color", "value": "<color>"}} - {{"action": "set_color", "value": [B, G, R]}} for ANY color, return BGR values as a JSON array (OpenCV BGR order, not RGB)
- {{"action": "set_image", "value": "<exact name from list>"}} pick the BEST matching image from the list above
- {{"action": "reset"}} go back to default
- {{"action": "unknown"}} only if intent is completely unclear

Rules:
- For city, country, region, place, or scene requests, always pick the best matching image from the list
- If multiple images match, pick the first/best one
- Use the EXACT name from the list as the value (no extension needed)
- "romania", "timis", "timisoara" should all match "timisoara" if that's the closest image
- If the user says a single color word (blue, red, green, teal, etc.), treat it as set_color
Examples (assuming list has "timisoara", "tokyo", "forest"):
- "tokyo" → {{"action": "set_image", "value": "tokyo"}}
- "romania" → {{"action": "set_image", "value": "timisoara"}}
- "make it blue" → {{"action": "set_color", "value": "blue"}}

User command: "{user_input}"
"""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        })
        text = response.json()["response"].strip()
        return json.loads(text)
    except Exception as e:
        print(f"LLM error: {e}")
        return {"action": "unknown"}
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DATA_FILE = BASE_DIR / "warehouse_data.json"

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
app = Flask(__name__)

def load_data():
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def system_prompt():
    return """You are a Warehouse Management AI Assistant.
Help users with warehouse inventory, product locations, stock levels,
rack/shelf information, warehouse capacity, incoming goods, outgoing goods,
low-stock alerts, reorder suggestions, and daily warehouse summaries.
Use the provided warehouse demo data. Do not invent data. If the data is
insufficient, clearly say so. Give simple, practical answers.

WAREHOUSE DATA:
""" + json.dumps(load_data(), indent=2)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "configured": client is not None, "model": MODEL})

@app.post("/api/chat")
def chat():
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is missing. Add it to your .env file."}), 500

    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])
    safe = [m for m in messages[-30:] if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)]

    if not safe:
        return jsonify({"error": "Please enter a message."}), 400

    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in safe]
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config={"system_instruction": system_prompt()}
        )
        return jsonify({"reply": response.text or "I could not generate a response."})
    except Exception as error:
        print("Gemini error:", repr(error))
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)

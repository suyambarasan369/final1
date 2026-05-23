import os
import json
import re

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except Exception:
    genai = None
    _GENAI_AVAILABLE = False

# Configure only if an API key is provided via environment variables
if _GENAI_AVAILABLE:
    _API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if _API_KEY:
        try:
            genai.configure(api_key=_API_KEY)
        except Exception:
            _GENAI_AVAILABLE = False


def analyze_text(prompt, detections=None):
    """Return a JSON string with: summary, points (list), performance_accuracy (float 0-100) and raw text.

    If the GenAI client isn't configured, returns a simple JSON explaining why.
    """
    if detections:
        prompt = f"{prompt}\n\nDetection summary:\n{detections}\n\nPlease provide a concise analysis with bullet points and a performance accuracy estimate (0-100%)."
    else:
        prompt = f"{prompt}\n\nPlease provide a concise analysis with bullet points and a performance accuracy estimate (0-100%)."

    if not _GENAI_AVAILABLE:
        return json.dumps({
            "summary": "GenAI not configured",
            "points": [],
            "performance_accuracy": None,
            "raw": "",
        })

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        # Use safety_settings to avoid MessageToDict compatibility issues
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            }
        ]
        try:
            response = model.generate_content(prompt, safety_settings=safety_settings)
        except Exception:
            # Fallback if safety_settings fails
            response = model.generate_content(prompt)
        
        text = ""
        try:
            text = response.text
        except Exception:
            # If .text fails, try accessing candidates
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        text = content.parts[0].text
            except Exception:
                pass
            
            if not text:
                text = str(response)

        # Extract bullet points (lines that start with -, • or numbered lists)
        points = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(('-', '•')) or re.match(r'^\d+\.', s):
                points.append(re.sub(r'^[-•\s\d\.]+', '', s).strip())

        # Find a percentage in the text (e.g., "85%" or "85 percent")
        perf = None
        m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if not m:
            m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*percent", text, re.IGNORECASE)
        if m:
            try:
                perf = float(m.group(1))
            except Exception:
                perf = None

        result = {
            "summary": text.splitlines()[0] if text else "Analysis complete",
            "points": points,
            "performance_accuracy": perf,
            "raw": text,
        }
        return json.dumps(result)
    except Exception as e:
        err_msg = str(e)
        return json.dumps({
            "summary": "Analysis complete but details unavailable" if "MessageToDict" in err_msg else f"Generation failed: {err_msg}",
            "points": [],
            "performance_accuracy": None,
            "raw": "",
        })
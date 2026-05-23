"""
Developer Agent
---------------
Generates self-contained HTML demos that render correctly in a sandboxed iframe.
React code is wrapped in a full Babel/CDN HTML document automatically.
Only called when user explicitly requests a demo.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session

from core.config import GROQ_API_KEY, GROQ_MODEL
from db.models import Milestone
from graph.state import AgentState

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)

DEVELOPER_SYSTEM = """### AGENT PERSONA
You are the Generation Engine UI Architect. You build clean, self-contained educational frontend demos.

### INPUT
- [CONCEPT]: {concept}
- [STACK]: {current_tech_stack}

### OUTPUT — return ONLY this raw JSON, no fences, no prose:
{
  "component_type": "react",
  "title": "Short descriptive title",
  "description": "One sentence explaining what this demonstrates.",
  "code": "THE FULL WORKING CODE AS A STRING — see rules below"
}

### CODE RULES
Write the code as a complete self-contained React component.
The code string must follow this exact template (fill in the actual logic):

import React, { useState, useEffect } from 'react';

function App() {
  // --- your state and logic here ---
  return (
    <div style={{ fontFamily: 'sans-serif', padding: '24px', background: '#fff', minHeight: '100vh' }}>
      {/* --- your JSX here --- */}
    </div>
  );
}

export default App;

RULES:
- Use only inline styles — no external CSS, no Tailwind classes.
- Make it visually rich: use colors, borders, cards, buttons, animations where relevant.
- The component must be INTERACTIVE — buttons, inputs, toggles, counters, something the user clicks.
- Add short inline comments explaining what each section does educationally.
- Demonstrate exactly ONE concept clearly. Do not overcomplicate.
- The code must actually work. No placeholders, no TODOs."""


def _wrap_react_in_html(react_code: str) -> str:
    """Wraps a React component string in a full Babel CDN HTML document for iframe rendering."""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="https://unpkg.com/react@18/umd/react.development.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
{react_code}

const rootEl = document.getElementById('root');
const root = ReactDOM.createRoot(rootEl);
root.render(<App />);
  </script>
</body>
</html>"""


async def run_developer(state: AgentState, db: Session, concept: str | None = None) -> dict:
    if not concept:
        node_id = state.get("active_node_id")
        if node_id:
            milestone = db.query(Milestone).filter(Milestone.id == node_id).first()
            concept = f"{milestone.title}: {milestone.description}" if milestone else None

    if not concept:
        concept = f"Introduction to {state.get('current_tech_stack', 'web development')}"

    tech = state.get("current_tech_stack") or "General Web Development"

    prompt = f"Generate an interactive educational React demo for this concept:\n{concept}"

    system = DEVELOPER_SYSTEM.replace("{concept}", concept).replace("{current_tech_stack}", tech)

    response = await _llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt)
    ])

    raw = response.content.strip()
    # Strip markdown fences if present
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                raw = stripped
                break

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If the model returned raw code instead of JSON, wrap it
        data = {
            "component_type": "react",
            "title"         : concept[:50],
            "description"   : f"Interactive demo for: {concept}",
            "code"          : raw
        }

    # Always pre-wrap the code into a full iframe-ready HTML document
    # so the frontend never has to do any transformation
    data["iframe_html"] = _wrap_react_in_html(data.get("code", ""))

    chat_message = (
        f"🛠️ **Demo ready: {data.get('title', 'Demo')}**\n\n"
        f"{data.get('description', '')}\n\n"
        "Check the **AI Class** tab to interact with it."
    )

    return {
        "generated_code": json.dumps(data),
        "messages"      : [{"role": "assistant", "content": chat_message}]
    }
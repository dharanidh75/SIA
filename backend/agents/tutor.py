"""
Tutor Agent — Gen UI Mode
--------------------------
Returns structured JSON lesson cards that the frontend renders as rich UI.
Only starts teaching when the user explicitly signals learning intent.
Casual messages get casual replies — no unsolicited lectures.
"""

from typing import AsyncGenerator
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from sqlalchemy.orm import Session

from core.config import GROQ_API_KEY, GROQ_MODEL
from db.models import UserSession, Milestone
from graph.state import AgentState

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.7)

# ── Intent signals that trigger a full Gen UI lesson ─────────────────────────
TEACH_TRIGGERS = [
    "let's start", "lets start", "start learning", "teach me", "begin",
    "what's next", "whats next", "next topic", "explain", "show me how",
    "i'm ready", "im ready", "go ahead", "start", "continue", "next"
]

def _is_teaching_request(user_input: str) -> bool:
    lowered = user_input.lower().strip()
    return any(trigger in lowered for trigger in TEACH_TRIGGERS)


def _build_casual_system() -> str:
    return """You are SIA, a friendly developer mentor.

No roadmap or lesson is active right now.

RULES:
- Be warm, casual, short. Max 2 sentences.
- If the user greets you, greet back and ask how you can help.
- If they mention wanting to learn something, encourage them to describe their goal.
- NEVER start teaching, never write code, never give a lesson.
- You are a companion, not a lecturer."""


def _build_chat_system(title: str, desc: str, day: int, done: bool, tech: str) -> str:
    return f"""You are SIA, a sharp developer mentor. You are in an active learning session.

CURRENT LESSON: {title}
DETAILS: {desc}
TECH: {tech} | DAY: {day} | DAY COMPLETE: {done}

CONVERSATIONAL MODE RULES:
- Answer questions about the current lesson naturally and concisely.
- If the user asks something off-topic, redirect warmly to the current lesson.
- If Day Complete is False and user asks to skip ahead, decline warmly.
- Keep replies under 200 words.
- Use markdown code blocks for any code.
- Do NOT produce JSON. This is a plain conversational reply.
- Greetings ("hey", "hello", "thanks") get a 1-sentence casual reply, nothing more."""


def _build_lesson_system(title: str, desc: str, day: int, tech: str, topics: list) -> str:
    topics_str = "\n".join(f"  - {t}" for t in topics) if topics else "  - Core concepts\n  - Practical application"
    return f"""You are SIA, an expert developer educator. You are delivering a structured lesson.

LESSON: {title}
DESCRIPTION: {desc}
TECH STACK: {tech}
DAY: {day}
TOPICS TO COVER:
{topics_str}

YOUR JOB: Produce a complete, rich, structured lesson covering ALL the topics above.
This must be the most thorough explanation the student has ever read on this topic.

OUTPUT FORMAT — return ONLY this raw JSON, no fences, no extra text:
{{
  "type": "lesson",
  "title": "{title}",
  "sections": [
    {{
      "kind": "concept",
      "heading": "Section heading",
      "body": "Clear explanation. Use real examples. Be thorough — 3-5 paragraphs if needed."
    }},
    {{
      "kind": "code",
      "heading": "Code example heading",
      "language": "javascript or python or html etc",
      "code": "// Full working code example with inline comments explaining each line",
      "explanation": "What this code demonstrates and why it matters."
    }},
    {{
      "kind": "tip",
      "heading": "Pro tip or common mistake",
      "body": "Practical advice a senior engineer would give."
    }}
  ],
  "challenge": {{
    "instruction": "A specific, achievable coding challenge tied to this lesson.",
    "hint": "One helpful hint without giving the answer away."
  }}
}}

MANDATORY RULES:
- Cover EVERY topic listed above — each must appear as its own section.
- Minimum 6 sections total (mix of concept, code, tip).
- Code sections must have REAL, RUNNABLE code — no pseudocode, no placeholders.
- The lesson must be self-contained. A student should understand the full topic after reading it.
- The challenge must require the student to actually write code."""


async def run_tutor_stream(
    state: AgentState, db: Session, user_input: str
) -> AsyncGenerator[str, None]:
    session_id = state.get("session_id", "")

    session_row = db.query(UserSession).filter(UserSession.id == session_id).first()

    day_completed     = session_row.day_completed      if session_row else False
    current_day       = session_row.current_day        if session_row else 1
    tech_stack        = session_row.current_tech_stack if session_row else ""
    active_node_id    = session_row.active_node_id     if session_row else None

    active_node_title = state.get("active_node_title", "") or ""
    active_node_desc  = ""
    lecture_topics    = []

    if active_node_id:
        milestone = db.query(Milestone).filter(Milestone.id == active_node_id).first()
        if milestone:
            active_node_title = milestone.title
            active_node_desc  = milestone.description

    # ── Route: no roadmap → casual only ──────────────────────────────────────
    if not active_node_title:
        system  = SystemMessage(content=_build_casual_system())
        history = [system, HumanMessage(content=user_input)]
        async for chunk in _llm.astream(history):
            if chunk.content: yield chunk.content
        return

    # ── Route: explicit teach trigger → Gen UI lesson JSON ───────────────────
    if _is_teaching_request(user_input):
        system = SystemMessage(content=_build_lesson_system(
            active_node_title, active_node_desc,
            current_day, tech_stack, lecture_topics
        ))
        history = [system, HumanMessage(content=f"Teach me: {active_node_title}")]
        full = []
        async for chunk in _llm.astream(history):
            if chunk.content:
                full.append(chunk.content)
                yield chunk.content

        # Validate JSON — if broken, yield a plain fallback marker
        raw = "".join(full).strip()
        if not raw.startswith("{"):
            yield "\n[PLAIN]"  # signal to frontend: render as markdown
        return

    # ── Route: conversational question about lesson ───────────────────────────
    system = SystemMessage(content=_build_chat_system(
        active_node_title, active_node_desc,
        current_day, day_completed, tech_stack
    ))

    history: list[BaseMessage] = [system]
    for msg in state.get("messages", []):
        if isinstance(msg, (HumanMessage, AIMessage)):
            history.append(msg)
        elif isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content", "")
            if role == "user":    history.append(HumanMessage(content=content))
            elif role == "assistant": history.append(AIMessage(content=content))

    history.append(HumanMessage(content=user_input))

    async for chunk in _llm.astream(history):
        if chunk.content: yield chunk.content


async def run_tutor_sync(state: AgentState, db: Session, user_input: str) -> dict:
    tokens = []
    async for token in run_tutor_stream(state, db, user_input):
        tokens.append(token)
    return {"messages": [{"role": "assistant", "content": "".join(tokens)}]}


def mark_day_complete(session_id: str, db: Session) -> dict:
    session_row = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session_row:
        return {"success": False, "message": "Session not found."}
    session_row.day_completed = True
    db.commit()
    return {"success": True, "message": "Day marked complete. Great work!"}


def advance_to_next_milestone(session_id: str, db: Session) -> dict:
    from db.models import Roadmap
    session_row = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session_row or not session_row.active_roadmap_id:
        return {"success": False, "message": "No active roadmap found."}

    if session_row.active_node_id:
        current = db.query(Milestone).filter(Milestone.id == session_row.active_node_id).first()
        if current: current.completed = True

    roadmap = db.query(Roadmap).filter(Roadmap.id == session_row.active_roadmap_id).first()
    if not roadmap:
        return {"success": False, "message": "Roadmap not found."}

    incomplete = [m for m in roadmap.milestones if not m.completed]
    if not incomplete:
        session_row.active_node_id = None
        db.commit()
        return {"success": True, "message": "🎉 You've completed the entire roadmap!", "roadmap_complete": True}

    nxt = incomplete[0]
    session_row.active_node_id = nxt.id
    session_row.current_day   += 1
    session_row.day_completed  = False
    db.commit()
    return {
        "success": True, "message": f"Moving to: **{nxt.title}**",
        "next_milestone_id": nxt.id, "next_milestone": nxt.title, "roadmap_complete": False
    }
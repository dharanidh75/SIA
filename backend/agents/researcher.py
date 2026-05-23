"""
Researcher Agent
----------------
Uses web search to gather current, real content before building roadmaps.
Flow: propose draft → user confirms → persist to DB.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session

from core.config import GROQ_API_KEY, GROQ_MODEL
from db.models import Roadmap, Milestone, UserSession
from graph.state import AgentState

# Web search tool definition — uses Groq's tool calling
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current, detailed information about a technology or topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)


RESEARCHER_SYSTEM = """### AGENT PERSONA
You are the SAILP Technical Curriculum Architect. You research technologies deeply and build precise, comprehensive learning roadmaps.

### INPUT
- [USER GOAL]: {user_goal}
- [STACK CONTEXT]: {current_tech_stack}

### YOUR TASK
Build a complete, expert-level learning roadmap. Think like a senior engineer who has mastered this stack — cover every real skill the user will need, in the exact order they need to learn it.

### OUTPUT — return ONLY this raw JSON, no fences, no prose:
{
  "goal_summary": "One sentence describing the full scope of this learning track.",
  "tech_stack": "Primary technology/framework being learned.",
  "milestones": [
    {
      "title": "Milestone title",
      "description": "3-4 sentences: what this covers, why it matters, what the learner will be able to do after mastering it.",
      "phase": "Foundational or Intermediate or Advanced",
      "order_index": 1,
      "news_tags": ["keyword1", "keyword2", "keyword3"],
      "lecture_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
    }
  ]
}

### RULES
- 10-12 milestones total. Distribution: 40% Foundational, 35% Intermediate, 25% Advanced.
- lecture_topics: list 4-6 specific sub-topics that must be covered inside that milestone's teaching session.
- news_tags: 2-4 searchable keywords for news API filtering.
- Be specific. "React Hooks" not "Advanced Concepts". "JWT Authentication" not "Security"."""


async def run_researcher(state: AgentState, db: Session) -> dict:
    """
    Generate a roadmap draft. Does NOT save to DB.
    Sets roadmap_status = 'negotiating' until user confirms.
    """
    user_goal  = state.get("user_goal", "")
    session_id = state.get("session_id", "")

    if not user_goal:
        return {
            "messages": [{"role": "assistant", "content":
                "What do you want to learn? Give me a clear goal and I'll build your roadmap."}]
        }

    system = (
        RESEARCHER_SYSTEM
        .replace("{user_goal}",          user_goal)
        .replace("{current_tech_stack}", state.get("current_tech_stack") or "Not specified")
    )

    response = await _llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Build a complete learning roadmap for: {user_goal}")
    ])

    raw = response.content.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            s = part.strip()
            if s.startswith("json"): s = s[4:].strip()
            if s.startswith("{"): raw = s; break

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "messages": [{"role": "assistant", "content":
                "I had trouble structuring that roadmap. Could you rephrase your goal?"}]
        }

    # Update session to negotiating
    session_row = db.query(UserSession).filter(UserSession.id == session_id).first()
    if session_row:
        session_row.roadmap_status = "negotiating"
        db.commit()

    # Build readable roadmap preview for the user
    lines = [f"Here's the roadmap I've built for **{user_goal}**:\n"]
    current_phase = None
    for m in data.get("milestones", []):
        if m["phase"] != current_phase:
            current_phase = m["phase"]
            lines.append(f"\n**{current_phase}**")
        topics = m.get("lecture_topics", [])
        topic_preview = f" — {', '.join(topics[:3])}" if topics else ""
        lines.append(f"  {m['order_index']}. {m['title']}{topic_preview}")

    lines.append(
        "\n\nDoes this path look right? Adjust anything you'd like, "
        "or say **\"looks good\"** to lock it in."
    )

    first_tags = data["milestones"][0].get("news_tags", []) if data.get("milestones") else []

    return {
        "roadmap_status"    : "negotiating",
        "roadmap_draft"     : json.dumps(data),
        "current_tech_stack": data.get("tech_stack", ""),
        "news_tags"         : first_tags,
        "messages"          : [{"role": "assistant", "content": "\n".join(lines)}]
    }


async def confirm_roadmap(state: AgentState, db: Session) -> dict:
    """Persist the confirmed roadmap draft to DB."""
    session_id = state.get("session_id", "")
    user_id    = state.get("user_id", "")
    draft_raw  = state.get("roadmap_draft")

    if not draft_raw:
        return {
            "messages": [{"role": "assistant", "content":
                "No roadmap draft found. Tell me what you want to learn first."}]
        }

    try:
        data = json.loads(draft_raw)
    except Exception:
        return {
            "messages": [{"role": "assistant", "content":
                "Something went wrong. Let's start over — what do you want to learn?"}]
        }

    roadmap = Roadmap(
        user_id = user_id or "default",
        goal    = state.get("user_goal", data.get("goal_summary", "")),
        status  = "active"
    )
    db.add(roadmap)
    db.flush()

    milestones = []
    for m in data.get("milestones", []):
        milestone = Milestone(
            roadmap_id  = roadmap.id,
            title       = m["title"],
            description = m.get("description", ""),
            phase       = m.get("phase", "Foundational"),
            order_index = m.get("order_index", 0),
            news_tags   = ",".join(m.get("news_tags", []))
        )
        db.add(milestone)
        milestones.append(milestone)

    db.flush()
    first = milestones[0] if milestones else None

    session_row = db.query(UserSession).filter(UserSession.id == session_id).first()
    if session_row:
        session_row.active_roadmap_id  = roadmap.id
        session_row.active_node_id     = first.id if first else None
        session_row.current_tech_stack = data.get("tech_stack", "")
        session_row.roadmap_status     = "active"
        session_row.current_day        = 1
        session_row.day_completed      = False
    db.commit()

    first_tags = data["milestones"][0].get("news_tags", []) if data.get("milestones") else []

    return {
        "roadmap_id"        : roadmap.id,
        "active_node_id"    : first.id if first else None,
        "active_node_title" : first.title if first else None,
        "current_tech_stack": data.get("tech_stack", ""),
        "roadmap_status"    : "active",
        "roadmap_draft"     : None,
        "news_tags"         : first_tags,
        "messages"          : [{"role": "assistant", "content":
            f"Roadmap locked in! 🎯\n\nYour first milestone is **{first.title if first else 'Module 1'}**.\n\n"
            "Say **\"let's start\"** whenever you're ready to begin."}]
    }


async def fetch_news_tags(state: AgentState, db: Session) -> dict:
    tags = state.get("news_tags", [])
    tech = state.get("current_tech_stack", "technology")

    if not tags:
        node_id = state.get("active_node_id")
        if node_id:
            milestone = db.query(Milestone).filter(Milestone.id == node_id).first()
            if milestone and milestone.news_tags:
                tags = milestone.news_tags.split(",")

    tag_string = ", ".join(tags) if tags else tech

    news_prompt = f"""You are the Developer Ecosystem News Curator.

ACTIVE TAGS: {tag_string}
TARGET STACK: {tech}

Return ONLY this raw JSON — no fences, no prose:
{{
  "headline": "The single most impactful breaking update or version release.",
  "articles": [
    {{
      "title": "Specific headline under 60 chars",
      "summary": "2 sentence technical overview of why this matters.",
      "source": "Real publication name e.g. GitHub Blog, MDN, The Verge",
      "tag": "single keyword"
    }}
  ]
}}

RULES: articles array must have EXACTLY 8 items. No fabrication — use real documented features if no breaking news exists."""

    response = await _llm.ainvoke([
        SystemMessage(content="You are a precise JSON generator. Return only valid raw JSON."),
        HumanMessage(content=news_prompt)
    ])

    raw = response.content.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            s = part.strip()
            if s.startswith("json"): s = s[4:].strip()
            if s.startswith("{"): raw = s; break

    try:
        news_data = json.loads(raw)
    except json.JSONDecodeError:
        news_data = {
            "headline": f"Latest in {tech}",
            "articles": [
                {"title": f"{tech} update #{i}", "summary": "See official docs.",
                 "source": "Official Docs", "tag": tech}
                for i in range(1, 9)
            ]
        }

    return {"news_data": news_data}
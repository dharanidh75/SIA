"""
SAILP Backend — FastAPI Entry Point
-------------------------------------
Endpoints:
  GET  /                        health check
  POST /session/init            create or retrieve a user session
  POST /chat/stream             main streaming chat endpoint (SSE)
  POST /chat                    non-streaming fallback
  GET  /roadmap/{session_id}    fetch full roadmap for the sidebar
  POST /roadmap/advance         mark milestone complete, advance to next
  POST /day/complete            mark current day as done
  GET  /news/{session_id}       fetch 3x3 news grid data
  GET  /demo/{session_id}       get latest generated code for iframe
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage

from db.session import init_db, get_db
from db.models import UserSession, User, Roadmap, Milestone
from graph.graph import build_graph
from graph.state import AgentState
from agents.tutor import mark_day_complete, advance_to_next_milestone
from agents.researcher import fetch_news_tags
from core.config import GROQ_API_KEY


# ── In-memory session state store ────────────────────────────────────────────
# Stores AgentState per session_id. In production, replace with Redis.
_session_states: dict[str, AgentState] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SAILP API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SessionInitRequest(BaseModel):
    user_name  : str = "Learner"
    session_id : str | None = None   # pass existing to resume

class ChatRequest(BaseModel):
    user_input : str
    session_id : str

class AdvanceRequest(BaseModel):
    session_id : str

class DayCompleteRequest(BaseModel):
    session_id : str

class DemoRequest(BaseModel):
    session_id : str
    concept    : str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_session_state(session_id: str, db: Session) -> AgentState:
    """
    Returns the in-memory AgentState for a session, hydrating from DB
    if not already loaded. MySQL is the source of truth.
    """
    if session_id in _session_states:
        return _session_states[session_id]

    # Hydrate from DB
    session_row = db.query(UserSession).filter(
        UserSession.id == session_id
    ).first()

    active_node_title = None
    if session_row and session_row.active_node_id:
        milestone = db.query(Milestone).filter(
            Milestone.id == session_row.active_node_id
        ).first()
        if milestone:
            active_node_title = milestone.title

    state: AgentState = {
        "messages"          : [],
        "session_id"        : session_id,
        "user_id"           : session_row.user_id if session_row else "default",
        "user_goal"         : None,
        "active_node_title" : active_node_title,
        "active_node_id"    : session_row.active_node_id    if session_row else None,
        "roadmap_id"        : session_row.active_roadmap_id if session_row else None,
        "current_tech_stack": session_row.current_tech_stack if session_row else "",
        "roadmap_status"    : session_row.roadmap_status    if session_row else "none",
        "current_day"       : session_row.current_day       if session_row else 1,
        "day_completed"     : session_row.day_completed     if session_row else False,
        "generated_code"    : None,
        "news_tags"         : [],
        "next_agent"        : None,
        "tool_call_id"      : None,
    }

    _session_states[session_id] = state
    return state


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "SAILP backend running", "version": "1.0.0"}


@app.post("/session/init")
def session_init(req: SessionInitRequest, db: Session = Depends(get_db)):
    """
    Create or resume a user session.
    Returns session_id the frontend must persist and send with every request.
    """
    import uuid

    # Resume existing session
    if req.session_id:
        session_row = db.query(UserSession).filter(
            UserSession.id == req.session_id
        ).first()
        if session_row:
            state = _get_or_create_session_state(req.session_id, db)
            return {
                "session_id"        : req.session_id,
                "user_id"           : session_row.user_id,
                "roadmap_status"    : session_row.roadmap_status,
                "current_tech_stack": session_row.current_tech_stack,
                "current_day"       : session_row.current_day,
                "day_completed"     : session_row.day_completed,
                "active_node_title" : state.get("active_node_title")
            }

    # Create new user + session
    user = User(name=req.user_name)
    db.add(user)
    db.flush()

    session_id  = str(uuid.uuid4())
    session_row = UserSession(
        id              = session_id,
        user_id         = user.id,
        roadmap_status  = "none",
        current_day     = 1,
        day_completed   = False
    )
    db.add(session_row)
    db.commit()

    # Prime in-memory state
    _get_or_create_session_state(session_id, db)

    return {
        "session_id"        : session_id,
        "user_id"           : user.id,
        "roadmap_status"    : "none",
        "current_tech_stack": "",
        "current_day"       : 1,
        "day_completed"     : False,
        "active_node_title" : None
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Main streaming endpoint.
    For tutor responses: streams tokens via SSE.
    For researcher/developer: runs synchronously and returns full response streamed.
    """
    state = _get_or_create_session_state(req.session_id, db)

    # Append user message to state
    state["messages"] = list(state.get("messages", [])) + [
        HumanMessage(content=req.user_input)
    ]

    # Build the graph with the current DB session
    graph = build_graph(db)

    async def generate():
        nonlocal state

        # Run the orchestrator to determine which agent to call
        # We run the full graph but intercept streaming for tutor
        result_state = await graph.ainvoke(state)

        # Extract the latest AI message from the result
        new_messages = result_state.get("messages", [])
        latest_ai_content = ""

        for msg in reversed(new_messages):
            if isinstance(msg, AIMessage):
                latest_ai_content = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                latest_ai_content = msg.get("content", "")
                break

        # Update in-memory state
        _session_states[req.session_id] = result_state

        # Sync updated state flags back to memory from result
        for field in ["roadmap_id", "active_node_id", "active_node_title",
                      "current_tech_stack", "roadmap_status", "news_tags",
                      "generated_code"]:
            if field in result_state:
                _session_states[req.session_id][field] = result_state[field]

        # Stream the response token by token (simulated for non-tutor agents,
        # real streaming for tutor via the tutor's own astream)
        if latest_ai_content:
            # Stream word by word for a smooth feel on researcher/developer responses
            words = latest_ai_content.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield f"data: {token}\n\n"

        # Send metadata as final event so frontend can update sidebar
        meta = {
            "roadmap_status"    : result_state.get("roadmap_status", "none"),
            "active_node_title" : result_state.get("active_node_title"),
            "current_tech_stack": result_state.get("current_tech_stack"),
            "generated_code"    : result_state.get("generated_code") is not None,
            "news_data"         : result_state.get("news_data")
        }
        yield f"data: [META]{json.dumps(meta)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/stream/tutor")
async def chat_stream_tutor(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Direct tutor streaming endpoint — bypasses orchestrator for pure token
    streaming. Used when the frontend already knows the intent is a chat message
    and wants true token-by-token streaming without the orchestrator overhead.
    """
    from agents.tutor import run_tutor_stream

    state = _get_or_create_session_state(req.session_id, db)

    async def generate():
        full_reply = []
        async for token in run_tutor_stream(state, db, req.user_input):
            full_reply.append(token)
            yield f"data: {token}\n\n"

        # Update state with the full reply
        full_text = "".join(full_reply)
        current_messages = list(state.get("messages", []))
        current_messages.append(HumanMessage(content=req.user_input))
        current_messages.append(AIMessage(content=full_text))
        _session_states[req.session_id] = {
            **state,
            "messages": current_messages
        }

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/roadmap/{session_id}")
def get_roadmap(session_id: str, db: Session = Depends(get_db)):
    """Returns the full roadmap for the sidebar."""
    session_row = db.query(UserSession).filter(
        UserSession.id == session_id
    ).first()

    if not session_row or not session_row.active_roadmap_id:
        return {"roadmap": None, "milestones": []}

    roadmap = db.query(Roadmap).filter(
        Roadmap.id == session_row.active_roadmap_id
    ).first()

    if not roadmap:
        return {"roadmap": None, "milestones": []}

    milestones = [
        {
            "id"          : m.id,
            "title"       : m.title,
            "description" : m.description,
            "phase"       : m.phase,
            "order_index" : m.order_index,
            "completed"   : m.completed,
            "is_active"   : m.id == session_row.active_node_id
        }
        for m in roadmap.milestones
    ]

    return {
        "roadmap_id"  : roadmap.id,
        "goal"        : roadmap.goal,
        "status"      : roadmap.status,
        "milestones"  : milestones,
        "active_node" : session_row.active_node_id
    }


@app.post("/roadmap/advance")
def advance_milestone(req: AdvanceRequest, db: Session = Depends(get_db)):
    """Mark current milestone complete, move to next."""
    result = advance_to_next_milestone(req.session_id, db)

    # Sync in-memory state
    if req.session_id in _session_states and result.get("success"):
        state = _session_states[req.session_id]
        _session_states[req.session_id] = {
            **state,
            "active_node_id"   : result.get("next_milestone_id"),
            "active_node_title": result.get("next_milestone"),
            "day_completed"    : False,
        }

    return result


@app.post("/day/complete")
def complete_day(req: DayCompleteRequest, db: Session = Depends(get_db)):
    """Mark the current study day as completed."""
    result = mark_day_complete(req.session_id, db)

    if req.session_id in _session_states and result.get("success"):
        _session_states[req.session_id]["day_completed"] = True

    return result


@app.get("/news/{session_id}")
async def get_news(session_id: str, db: Session = Depends(get_db)):
    """Fetch 3x3 news grid data for the news view."""
    state = _get_or_create_session_state(session_id, db)

    # Pull news tags from active milestone if not in state
    tags = state.get("news_tags", [])
    if not tags and state.get("active_node_id"):
        milestone = db.query(Milestone).filter(
            Milestone.id == state["active_node_id"]
        ).first()
        if milestone and milestone.news_tags:
            tags = milestone.news_tags.split(",")
            _session_states[session_id]["news_tags"] = tags

    result = await fetch_news_tags(state, db)
    return result.get("news_data", {})


@app.post("/demo/generate")
async def generate_demo(req: DemoRequest, db: Session = Depends(get_db)):
    """On-demand developer agent call — returns generated code JSON."""
    from agents.developer import run_developer

    state  = _get_or_create_session_state(req.session_id, db)
    result = await run_developer(state, db, req.concept)

    if req.session_id in _session_states:
        _session_states[req.session_id]["generated_code"] = result.get("generated_code")

    return {
        "generated_code": result.get("generated_code"),
        "message"       : result.get("messages", [{}])[0].get("content", "")
    }


@app.get("/demo/{session_id}")
def get_demo(session_id: str):
    """Return the latest generated code for the iframe."""
    state = _session_states.get(session_id, {})
    raw   = state.get("generated_code")
    if not raw:
        return {"generated_code": None}
    try:
        return {"generated_code": json.loads(raw)}
    except Exception:
        return {"generated_code": None}


@app.get("/session/{session_id}/state")
def get_session_state(session_id: str, db: Session = Depends(get_db)):
    """Returns current state flags for the frontend to sync UI."""
    session_row = db.query(UserSession).filter(
        UserSession.id == session_id
    ).first()

    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")

    state = _session_states.get(session_id, {})

    return {
        "session_id"        : session_id,
        "roadmap_status"    : session_row.roadmap_status,
        "current_tech_stack": session_row.current_tech_stack,
        "current_day"       : session_row.current_day,
        "day_completed"     : session_row.day_completed,
        "active_node_title" : state.get("active_node_title"),
        "active_node_id"    : session_row.active_node_id,
    }
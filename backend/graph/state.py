from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── User / Session identity ───────────────────────────────────────────────
    session_id : str
    user_id    : str

    # ── Roadmap context ───────────────────────────────────────────────────────
    user_goal          : Optional[str]
    active_node_title  : Optional[str]   # title of the current milestone
    active_node_id     : Optional[str]   # DB id of the current milestone
    roadmap_id         : Optional[str]

    # ── State flags (mirrors UserSession DB row) ──────────────────────────────
    current_tech_stack : Optional[str]
    roadmap_status     : str             # "none" | "negotiating" | "active"
    current_day        : int
    day_completed      : bool

    # ── Agent outputs ─────────────────────────────────────────────────────────
    generated_code     : Optional[str]   # Developer agent output
    news_tags          : list[str]       # Tags for news feed filtering
    roadmap_draft      : Optional[str]   # JSON string of unconfirmed roadmap draft

    # ── Routing ───────────────────────────────────────────────────────────────
    next_agent         : Optional[str]   # which agent node to run next
    tool_call_id       : Optional[str]   # tracks the active LLM tool call
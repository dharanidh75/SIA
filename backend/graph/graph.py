"""
LangGraph Orchestrator Graph
-----------------------------
Architecture:
  User message
      ↓
  [orchestrator_node]  ← LLM with 3 tools defined; picks which agent to call
      ↓
  [researcher_node | tutor_node | developer_node]
      ↓
  [END]

The LLM itself decides routing via tool calling — no manual if/else intent
classification. Tool definitions describe each agent's purpose clearly so
the LLM can make the right call.
"""

import json
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from core.config import GROQ_API_KEY, GROQ_MODEL
from graph.state import AgentState
from agents.researcher import run_researcher, confirm_roadmap, fetch_news_tags
from agents.tutor import run_tutor_sync
from agents.developer import run_developer


# ── Orchestrator LLM ─────────────────────────────────────────────────────────
_orchestrator_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# ── Tool Definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_researcher",
            "description": (
                "Call this when the user wants to: set a learning goal, "
                "build a roadmap, learn a new technology, choose a tech stack, "
                "or ask 'what should I learn'. Also call this to fetch fresh "
                "industry news for the user's current tech stack."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The user's learning goal extracted from their message."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["roadmap", "news"],
                        "description": (
                            "'roadmap' to generate a learning plan, "
                            "'news' to fetch industry updates."
                        )
                    }
                },
                "required": ["mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_roadmap",
            "description": (
                "Call this ONLY when the user explicitly confirms or approves the "
                "roadmap that was just shown to them. Triggers include: "
                "'looks good', 'confirm', 'yes', 'let's go', 'that works', "
                "'lock it in', 'save it', or any clear approval of the proposed plan. "
                "Do NOT call this for a new learning goal — use run_researcher for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tutor",
            "description": (
                "Call this for any learning question, concept explanation, "
                "code help, follow-up question, general study conversation, "
                "greetings, or small talk. This is the default fallback agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The user's message to send to the tutor."
                    }
                },
                "required": ["user_input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_developer",
            "description": (
                "Call this when the user explicitly requests a code demo, "
                "visual example, interactive component, or says 'show me', "
                "'generate a demo', 'build an example', or 'Topic of the Day'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": (
                            "The specific concept or component to demonstrate. "
                            "Leave empty to use the active roadmap milestone."
                        )
                    }
                },
                "required": []
            }
        }
    }
]


# ── Graph Nodes ──────────────────────────────────────────────────────────────

async def orchestrator_node(state: AgentState, db: Session) -> AgentState:
    """
    The central routing node. Calls the LLM with tool definitions.
    The LLM returns a tool_call which determines which agent node runs next.

    NOTE: The system prompt is built as a plain f-string — NOT via .format().
    This avoids KeyError from curly braces inside the JSON examples in the prompt.
    """
    state_context = (
        f"- Tech Stack    : {state.get('current_tech_stack') or 'Not set'}\n"
        f"- Roadmap Status: {state.get('roadmap_status', 'none')}\n"
        f"- Active Lesson : {state.get('active_node_title') or 'None'}\n"
        f"- Day Index     : {state.get('current_day', 1)}\n"
        f"- Day Completed : {state.get('day_completed', False)}"
    )

    system_prompt = f"""### SYSTEM ROLE & IDENTITY
You are SIA, the Central AI Workspace Router and Orchestrator for SAILP (Self-Improving Agentic Learning Platform). You do not directly formulate educational code, build roadmaps, or draft news items. Your solitary objective is to ingest data chunks, classify intents, and call exactly ONE matching structural tool.

---

### CHUNK 1: PLATFORM STATE SNAPSHOT (INJECTED CONTEXT)
Evaluate these parameters with every incoming turn before reading the user's string:
{state_context}

---

### CHUNK 2: CENTRAL INTENT ROUTING MATRIX
You are restricted to executing one of the three tools below based on the following precise matching triggers. If multiple conditions look relevant, prioritize top-down rules:

1. TARGET: run_researcher
   - TRIGGER A (Roadmap): User wants to establish a goal, plan a tech career, choose a technology, or explicitly says "I want to learn [X]".
     ACTION: Call run_researcher with parameters -> mode="roadmap", goal="<user query>"
   - TRIGGER B (News): User requests real-time industry updates, ecosystem developments, or tab modifications to the "news" panel.
     ACTION: Call run_researcher with parameters -> mode="news"

2. TARGET: run_developer
   - TRIGGER: User requests a functional code block, visual demo, interactive container, sandbox implementation, or triggers a "Topic of the Day" lesson.
     ACTION: Call run_developer with parameters -> concept="<extracted technical theme or leave empty for active milestone>"

3. TARGET: run_tutor (DEFAULT FALLBACK GATEWAY)
   - TRIGGER: Any specific conceptual questions, debugging help, conversational small talk, greeting, or ambiguous conversational items.
     ACTION: Call run_tutor with parameters -> user_input="<unaltered user message>"

---

### CHUNK 3: CRITICAL OPERATIONAL GUARDRAILS
- ZERO PLAIN TEXT RULE: Never emit plain text responses to the workspace window. Every path must resolve to a valid tool function call.
- SMALL TALK PROCESSING: Do not process greeting text directly. Route greetings straight to run_tutor.
- AMBIGUITY HANDLING: If a query contains completely corrupt, mixed, or unknowable requests, execute run_tutor to allow the tutor persona to re-scope the user conversation loop.
"""

    system = SystemMessage(content=system_prompt)

    # Get the latest user message
    latest_user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_msg = msg.get("content", "")
            break

    llm_with_tools = _orchestrator_llm.bind_tools(
        tools=TOOLS,
        tool_choice="required"   # force the LLM to always pick a tool
    )

    response = await llm_with_tools.ainvoke(
        [system, HumanMessage(content=latest_user_msg)]
    )

    # Extract the tool call
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        return {
            **state,
            "next_agent"  : tool_call["name"],
            "tool_call_id": tool_call.get("id", ""),
            "_tool_args"  : tool_call.get("args", {})
        }

    # Fallback — shouldn't happen with tool_choice="required"
    return {**state, "next_agent": "run_tutor",
            "_tool_args": {"user_input": latest_user_msg}}


async def researcher_node(state: AgentState, db: Session) -> AgentState:
    args = state.get("_tool_args", {}) or {}
    mode = args.get("mode", "roadmap")

    if mode == "news":
        result = await fetch_news_tags(state, db)
        return {**state, **result, "_tool_args": {}}

    # Pull goal from tool args first, then fall back to the raw user message
    goal = args.get("goal", "").strip()
    if not goal:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                goal = msg.content.strip()
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                goal = msg.get("content", "").strip()
                break

    # Build an updated state copy with the goal injected — do NOT mutate in place
    updated_state = {**state, "user_goal": goal}

    result = await run_researcher(updated_state, db)
    return {**updated_state, **result, "_tool_args": {}}


async def confirm_node(state: AgentState, db: Session) -> AgentState:
    result = await confirm_roadmap(state, db)
    return {**state, **result, "_tool_args": {}}


async def tutor_node(state: AgentState, db: Session) -> AgentState:
    args       = state.get("_tool_args", {})
    user_input = args.get("user_input", "")

    if not user_input:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

    result = await run_tutor_sync(state, db, user_input)
    return {**state, **result, "_tool_args": {}}


async def developer_node(state: AgentState, db: Session) -> AgentState:
    args    = state.get("_tool_args", {})
    concept = args.get("concept", None)
    result  = await run_developer(state, db, concept)
    return {**state, **result, "_tool_args": {}}


# ── Conditional edge ──────────────────────────────────────────────────────────

def route_after_orchestrator(
    state: AgentState
) -> Literal["researcher_node", "confirm_node", "tutor_node", "developer_node"]:
    next_agent = state.get("next_agent", "run_tutor")
    mapping = {
        "run_researcher" : "researcher_node",
        "confirm_roadmap": "confirm_node",
        "run_tutor"      : "tutor_node",
        "run_developer"  : "developer_node"
    }
    return mapping.get(next_agent, "tutor_node")


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(db: Session) -> StateGraph:
    graph = StateGraph(AgentState)

    async def _orchestrator(state):
        return await orchestrator_node(state, db)

    async def _researcher(state):
        return await researcher_node(state, db)

    async def _confirm(state):
        return await confirm_node(state, db)

    async def _tutor(state):
        return await tutor_node(state, db)

    async def _developer(state):
        return await developer_node(state, db)

    graph.add_node("orchestrator_node", _orchestrator)
    graph.add_node("researcher_node",   _researcher)
    graph.add_node("confirm_node",      _confirm)
    graph.add_node("tutor_node",        _tutor)
    graph.add_node("developer_node",    _developer)

    graph.set_entry_point("orchestrator_node")

    graph.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        {
            "researcher_node": "researcher_node",
            "confirm_node"   : "confirm_node",
            "tutor_node"     : "tutor_node",
            "developer_node" : "developer_node"
        }
    )

    graph.add_edge("researcher_node", END)
    graph.add_edge("confirm_node",    END)
    graph.add_edge("tutor_node",      END)
    graph.add_edge("developer_node",  END)

    return graph.compile()
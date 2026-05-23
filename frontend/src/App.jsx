import { useState, useRef, useEffect } from 'react'
import './App.css';

const API = "http://localhost:8000";

// ── Session ID persisted in localStorage ─────────────────────────────────────
function getOrCreateSessionId() {
  const stored = localStorage.getItem("sailp_session_id");
  if (stored) return stored;
  // Will be replaced by the real session_id from /session/init
  return null;
}

function App() {
  const [view, setView]               = useState('home');
  const [query, setQuery]             = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionId, setSessionId]     = useState(null);
  const [sessionState, setSessionState] = useState({
    roadmap_status    : "none",
    current_tech_stack: "",
    current_day       : 1,
    day_completed     : false,
    active_node_title : null,
  });

  const [messages, setMessages] = useState([
    { role: "ai", content: "Hey! I'm SIA. What do you want to learn today?" }
  ]);

  // Roadmap sidebar
  const [roadmap, setRoadmap]   = useState(null);
  const [milestones, setMilestones] = useState([]);

  // News grid
  const [newsData, setNewsData]   = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);

  // Demo iframe
  const [demoCode, setDemoCode] = useState(null);

  const textareaRef = useRef(null);
  const bottomRef   = useRef(null);

  // ── Auto scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Session init on mount ──────────────────────────────────────────────────
  useEffect(() => {
    initSession();
  }, []);

  const initSession = async () => {
    const existing = getOrCreateSessionId();
    try {
      const res  = await fetch(`${API}/session/init`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({
          user_name  : "Learner",
          session_id : existing
        })
      });
      const data = await res.json();
      const sid  = data.session_id;
      localStorage.setItem("sailp_session_id", sid);
      setSessionId(sid);
      setSessionState({
        roadmap_status    : data.roadmap_status,
        current_tech_stack: data.current_tech_stack,
        current_day       : data.current_day,
        day_completed     : data.day_completed,
        active_node_title : data.active_node_title,
      });

      // If active roadmap, load sidebar
      if (data.roadmap_status === "active") {
        fetchRoadmap(sid);
      }
    } catch (e) {
      console.error("Session init failed:", e);
      // Offline fallback — still usable in basic chat mode
      const fallback = "local_" + Date.now();
      setSessionId(fallback);
    }
  };

  const fetchRoadmap = async (sid) => {
    try {
      const res  = await fetch(`${API}/roadmap/${sid || sessionId}`);
      const data = await res.json();
      if (data.roadmap_id) {
        setRoadmap(data);
        setMilestones(data.milestones || []);
      }
    } catch (e) {
      console.error("Roadmap fetch failed:", e);
    }
  };

  const fetchNews = async () => {
    if (!sessionId) return;
    setNewsLoading(true);
    try {
      const res  = await fetch(`${API}/news/${sessionId}`);
      const data = await res.json();
      setNewsData(data);
    } catch (e) {
      console.error("News fetch failed:", e);
    }
    setNewsLoading(false);
  };

  // Fetch news when switching to news tab
  useEffect(() => {
    if (view === 'news' && !newsData && sessionId) {
      fetchNews();
    }
  }, [view, sessionId]);

  // ── Send message ───────────────────────────────────────────────────────────
  const sendMessage = async () => {
    if (!query.trim() || chatLoading || !sessionId) return;

    const userText = query.trim();
    setQuery("");
    setChatLoading(true);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    setMessages(prev => [...prev, { role: "user", content: userText }]);
    setMessages(prev => [...prev, { role: "ai", content: "" }]);

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ user_input: userText, session_id: sessionId })
      });

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text  = decoder.decode(value);
        const lines = text.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const token = line.replace("data: ", "");

            // Handle metadata packet
            if (token.startsWith("[META]")) {
              try {
                const meta = JSON.parse(token.replace("[META]", ""));
                // Update session state from meta
                setSessionState(prev => ({
                  ...prev,
                  roadmap_status    : meta.roadmap_status    ?? prev.roadmap_status,
                  active_node_title : meta.active_node_title ?? prev.active_node_title,
                  current_tech_stack: meta.current_tech_stack ?? prev.current_tech_stack,
                }));
                // If roadmap was just created, reload sidebar
                if (meta.roadmap_status === "active") {
                  fetchRoadmap(sessionId);
                }
                // If developer agent ran, load demo
                if (meta.generated_code) {
                  loadDemo();
                }
                // If news data came back, store it
                if (meta.news_data) {
                  setNewsData(meta.news_data);
                }
              } catch (_) {}
              continue;
            }

            if (token) {
              await new Promise(r => setTimeout(r, 18));
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role   : "ai",
                  content: updated[updated.length - 1].content + token
                };
                return updated;
              });
            }
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role   : "ai",
          content: "⚠️ Can't reach SIA backend. Is FastAPI running?"
        };
        return updated;
      });
    }

    setChatLoading(false);
  };

  const loadDemo = async () => {
    if (!sessionId) return;
    try {
      const res  = await fetch(`${API}/demo/${sessionId}`);
      const data = await res.json();
      if (data.generated_code) {
        setDemoCode(data.generated_code);
      }
    } catch (e) {
      console.error("Demo fetch failed:", e);
    }
  };

  const handleAdvance = async () => {
    if (!sessionId) return;
    try {
      const res  = await fetch(`${API}/roadmap/advance`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ session_id: sessionId })
      });
      const data = await res.json();
      if (data.success) {
        fetchRoadmap(sessionId);
        setMessages(prev => [...prev, { role: "ai", content: data.message }]);
      }
    } catch (e) {
      console.error("Advance failed:", e);
    }
  };

  const handleDayComplete = async () => {
    if (!sessionId) return;
    try {
      await fetch(`${API}/day/complete`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ session_id: sessionId })
      });
      setSessionState(prev => ({ ...prev, day_completed: true }));
    } catch (e) {
      console.error("Day complete failed:", e);
    }
  };

  // Re-focus textarea
  useEffect(() => {
    if (!chatLoading && textareaRef.current) textareaRef.current.focus();
  }, [chatLoading]);

  // ── Render demo code in iframe ─────────────────────────────────────────────
  const getDemoSrcDoc = () => {
    if (!demoCode) return "";
    if (demoCode.component_type === "html") return demoCode.code;
    // React: wrap in a simple babel CDN loader
    return `<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>body{margin:0;background:#fff;font-family:sans-serif;}</style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    ${demoCode.code}
    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(React.createElement(App || (() => <div>Component loaded</div>)));
  </script>
</body>
</html>`;
  };

  // ── Phase colour ───────────────────────────────────────────────────────────
  const phaseColour = {
    Foundational: "#9e0000",
    Intermediate: "#b35c00",
    Advanced    : "#1a5c1a"
  };

  return (
    <>
      <nav>
        <h1 className='header_name'>SIA</h1>
        <center>
          <div className='center_option'>
            {['AI class', 'Home', 'News'].map((name) => {
              const key      = name.toLowerCase().replace(' ', '-');
              const isActive = view === key;
              return (
                <h1
                  key={name}
                  className={`option_name${isActive ? ' option_name--active' : ''}`}
                  onClick={() => setView(key)}
                >
                  {name}
                </h1>
              );
            })}
          </div>
        </center>
        <h1 className='profile'>P</h1>
      </nav>

      {/* ── Headlines Banner ── */}
      <div className='headlines'>
        <p className='headline_text'>
          {newsData?.headline
            ? `📰 ${newsData.headline}`
            : sessionState.active_node_title
              ? `📚 Currently Learning: ${sessionState.active_node_title}`
              : "Tell SIA what you want to learn to get started."}
        </p>
      </div>

      {/* ── HOME VIEW ── */}
      {view === 'home' && (
        <div className='main_container'>

          {/* ROADMAP SIDEBAR */}
          <div className='roadmap_container'>
            <h1 className='roadmap_box'>ROADMAP</h1>

            {milestones.length === 0 ? (
              <p className='roadmap_empty'>
                Tell SIA your learning goal in the chat to generate your roadmap.
              </p>
            ) : (
              <div className='roadmap_list'>
                {milestones.map((m) => (
                  <div
                    key={m.id}
                    className={`milestone_item ${m.is_active ? 'milestone_active' : ''} ${m.completed ? 'milestone_done' : ''}`}
                  >
                    <span
                      className='milestone_phase_dot'
                      style={{ background: phaseColour[m.phase] || "#444" }}
                    />
                    <span className='milestone_title'>{m.title}</span>
                    {m.completed && <span className='milestone_check'>✓</span>}
                  </div>
                ))}

                {sessionState.day_completed && (
                  <button className='advance_btn' onClick={handleAdvance}>
                    Next Milestone →
                  </button>
                )}
              </div>
            )}
          </div>

          {/* CHATBOT */}
          <div className='chatbot_container'>
            <div className='chatbot_header'>
              <h1 className='chatbot_box'>CHATBOT</h1>
              {sessionState.active_node_title && (
                <div className='day_status'>
                  <span className='day_label'>Day {sessionState.current_day}</span>
                  {!sessionState.day_completed ? (
                    <button className='day_complete_btn' onClick={handleDayComplete}>
                      Mark Done ✓
                    </button>
                  ) : (
                    <span className='day_done_badge'>Day Complete ✓</span>
                  )}
                </div>
              )}
            </div>

            <div className="messages_area">
              {messages.map((msg, i) => (
                <div key={i} className={`bubble ${msg.role}`}>
                  {msg.content}
                </div>
              ))}
              {chatLoading && (
                <div className="bubble ai loading">
                  <span /><span /><span />
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            <center>
              <div className="chat_input_wrapper">
                <textarea
                  ref={textareaRef}
                  autoFocus
                  className="query_box"
                  placeholder={chatLoading ? "SIA is thinking..." : "chat with SIA..."}
                  value={query}
                  rows={1}
                  disabled={chatLoading}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    e.target.style.height = "auto";
                    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                />
                <div className="chat_toolbar">
                  <div />
                  <button
                    className="send_btnf"
                    onClick={sendMessage}
                    disabled={chatLoading || !query.trim()}
                  >
                    {chatLoading ? "..." : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2.5"
                        strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="19" x2="12" y2="5" />
                        <polyline points="5 12 12 5 19 12" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </center>
          </div>
        </div>
      )}

      {/* ── AI CLASS VIEW ── */}
      {view === 'ai-class' && (
        <div className='full_view'>
          {demoCode ? (
            <div className='demo_wrapper'>
              <div className='demo_header'>
                <h2 className='demo_title'>{demoCode.title}</h2>
                <p className='demo_desc'>{demoCode.description}</p>
                <button
                  className='start_learning_btn'
                  onClick={() => {
                    const concept = prompt("Enter concept to demo:");
                    if (concept && sessionId) {
                      fetch(`${API}/demo/generate`, {
                        method : "POST",
                        headers: { "Content-Type": "application/json" },
                        body   : JSON.stringify({ session_id: sessionId, concept })
                      }).then(r => r.json()).then(d => {
                        if (d.generated_code) setDemoCode(d.generated_code);
                      });
                    }
                  }}
                >
                  New Demo
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2.5"
                    strokeLinecap="round" strokeLinejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12" />
                    <polyline points="12 5 19 12 12 19" />
                  </svg>
                </button>
              </div>
              <iframe
                className='demo_iframe'
                srcDoc={getDemoSrcDoc()}
                sandbox="allow-scripts"
                title="SIA Demo"
              />
            </div>
          ) : (
            <div className='demo_placeholder'>
              <p className='news_placeholder_text'>No demo loaded yet.</p>
              <span className='news_placeholder_sub'>
                Ask SIA to "generate a demo" or "show me an example" in the chat.
              </span>
              <button
                className='start_learning_btn'
                style={{ marginTop: "20px" }}
                onClick={() => setView('home')}
              >
                Go to Chat
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2.5"
                  strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── NEWS VIEW ── */}
      {view === 'news' && (
        <div className='full_view news_view'>
          {newsLoading ? (
            <div className='news_loading'>
              <div className="bubble ai loading" style={{alignSelf:'center'}}>
                <span /><span /><span />
              </div>
              <p style={{color:'#555', marginTop:'12px'}}>Fetching latest updates...</p>
            </div>
          ) : newsData ? (
            <div className='news_grid_wrapper'>
              {/* Headline row */}
              <div className='news_headline_card'>
                <span className='news_headline_tag'>TOP STORY</span>
                <p className='news_headline_main'>{newsData.headline}</p>
              </div>

              {/* 3x3 grid (8 articles) */}
              <div className='news_grid'>
                {(newsData.articles || []).slice(0, 8).map((article, i) => (
                  <div key={i} className='news_card'>
                    <span className='news_card_tag'>{article.tag}</span>
                    <p className='news_card_title'>{article.title}</p>
                    <p className='news_card_summary'>{article.summary}</p>
                    <span className='news_card_source'>{article.source}</span>
                  </div>
                ))}
              </div>

              <button className='news_refresh_btn' onClick={fetchNews}>
                Refresh
              </button>
            </div>
          ) : (
            <div className='news_placeholder'>
              <div className='news_placeholder_icon'>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                  stroke="#9e0000" strokeWidth="1.5"
                  strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2" />
                  <path d="M18 14h-8M15 18h-5M10 6h8v4h-8V6Z" />
                </svg>
              </div>
              <p className='news_placeholder_text'>News not loaded yet.</p>
              <span className='news_placeholder_sub'>
                Generate your roadmap first, then news will be tailored to your stack.
              </span>
              <button
                className='start_learning_btn'
                style={{ marginTop: "20px" }}
                onClick={fetchNews}
              >
                Load News
              </button>
            </div>
          )}
        </div>
      )}
    </>
  );
}

export default App;
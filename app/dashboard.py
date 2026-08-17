"""
IdemAI Dashboard — Streamlit metrics & Aziz Agent console.

Intentionally scoped to:
  1. Dataset metrics (hours, quality, per-language breakdown).
  2. Script chunk browser (read-only, per language).
  3. Aziz Agent chat console (text commands only here; voice goes via voice_processor).

Recording of training audio is deliberately NOT here — use recording_server.py
(FastAPI + browser MediaRecorder) for that, then point the dashboard at the manifest.

Run: streamlit run app/dashboard.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import streamlit as st
import streamlit.components.v1

# Data lives outside OneDrive to avoid sync-lock conflicts
_DEFAULT_MANIFEST = Path.home() / "idem-ai-data" / "master_manifest.jsonl"

st.set_page_config(
    page_title="IdemAI Core",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    manifest_path = st.text_input(
        "Manifest path",
        value=str(_DEFAULT_MANIFEST),
        help="Path to master_manifest.jsonl",
    )
    _default_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    )
    api_key = st.text_input(
        "API key (Gemini or Anthropic)",
        value=_default_key,
        type="password",
        help="Gemini: AIza… key from aistudio.google.com (free). Anthropic: sk-ant-… key.",
    )
    language_filter = st.selectbox(
        "Language filter (scripts view)",
        options=["efi", "ibb", "en_NG"],
        format_func=lambda c: {"efi": "Efik", "ibb": "Ibibio", "en_NG": "Nigerian English"}[c],
    )

# ---------------------------------------------------------------------------
# Metrics tab
# ---------------------------------------------------------------------------
def render_metrics() -> None:
    from engine.audio_pipeline import manifest_stats

    st.subheader("📊 Dataset Metrics")
    try:
        stats = manifest_stats(Path(manifest_path))
    except Exception as e:
        st.error(f"Could not read manifest: {e}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total entries", stats["total_entries"])
    col2.metric("Clean entries", stats["clean_entries"])
    col3.metric("Total hours", f"{stats['total_hours']:.2f} h")
    col4.metric("Clean hours", f"{stats['clean_hours']:.2f} h")

    if stats["total_entries"] > 0:
        st.caption(
            f"⚠️ {stats['clipped']} clipped  |  "
            f"🔇 {stats['low_snr']} low-SNR (<15 dB)"
        )

    if stats["by_language_hours"]:
        st.markdown("**Hours by language (clean)**")
        lang_names = {"efi": "Efik", "ibb": "Ibibio", "en_NG": "Nigerian English"}
        for code, hours in sorted(stats["by_language_hours"].items()):
            label = lang_names.get(code, code)
            st.progress(
                min(hours / max(stats["clean_hours"], 0.001), 1.0),
                text=f"{label}: {hours:.3f} h",
            )
    else:
        st.info("No data in manifest yet. Start recording to see metrics here.")


# ---------------------------------------------------------------------------
# Script browser tab
# ---------------------------------------------------------------------------
def render_scripts() -> None:
    from engine.script_generator import mock_scripts

    st.subheader(f"📝 Script Chunks — {language_filter.upper()}")
    n = st.slider("Number of chunks", min_value=3, max_value=20, value=8)

    if st.button("Generate"):
        with st.spinner("Generating…"):
            try:
                chunks = mock_scripts(language_filter, n)
                for i, c in enumerate(chunks, 1):
                    badge = "✅" if c.in_target_range else "⚠️"
                    st.markdown(
                        f"**{i}.** {badge} `{c.word_count} words` — {c.text}"
                    )
            except Exception as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Aziz console tab
# ---------------------------------------------------------------------------
def render_aziz() -> None:
    st.subheader("🤖 Aziz Agent Console")

    if not api_key:
        st.warning(
            "**Aziz needs an API key.** Enter it in the sidebar, or add one to `.env` and restart.\n\n"
            "**Option A — Gemini (free):** get a key at [aistudio.google.com](https://aistudio.google.com) "
            "→ add `GOOGLE_API_KEY=AIza...` to `.env`\n\n"
            "**Option B — Anthropic:** add `ANTHROPIC_API_KEY=sk-ant-...` to `.env`"
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested commands the user can click
    st.caption("Try asking:")
    cols = st.columns(3)
    suggestions = [
        "Generate 5 Efik scripts",
        "Show dataset status",
        "List files in Downloads",
    ]
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state["_suggested"] = suggestion

    user_input = st.chat_input(
        "e.g. 'generate 5 Efik scripts' · 'validate Ibibio text: …' · 'show dataset status'"
    )
    # Handle suggestion button clicks
    if "_suggested" in st.session_state:
        user_input = st.session_state.pop("_suggested")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Aziz is thinking…"):
                try:
                    import json
                    from agent.aziz_orchestrator import AzizOrchestrator
                    aziz = AzizOrchestrator(
                        api_key=api_key,
                        config={"manifest_path": manifest_path},
                    )
                    response = aziz.run(user_input)
                    backend_label = "Gemini" if aziz.backend == "gemini" else "Claude"

                    # Human-readable summary for common tools
                    result = response.result
                    summary = ""
                    if response.tool_name == "generate_scripts":
                        chunks = result.get("chunks", [])
                        summary = f"Generated {len(chunks)} script chunk(s):\n\n"
                        for i, c in enumerate(chunks, 1):
                            summary += f"**{i}.** {c['text']}  `{c['word_count']} words`\n\n"
                    elif response.tool_name == "get_status":
                        summary = (
                            f"📦 **{result.get('total_entries', 0)}** total entries · "
                            f"✅ **{result.get('clean_entries', 0)}** clean · "
                            f"⏱ **{result.get('clean_hours', 0):.3f} hours**"
                        )
                    elif response.tool_name == "browse_local_files":
                        files = result.get("files", [])
                        if files:
                            summary = f"Found {len(files)} file(s):\n" + "\n".join(f"- `{f}`" for f in files[:20])
                        else:
                            summary = f"No files found in `{result.get('directory', '.')}`."
                    elif response.tool_name == "validate_text":
                        if result.get("is_valid"):
                            summary = f"✅ Text is valid for `{result.get('language', '')}`."
                        else:
                            errs = result.get("errors", [])
                            summary = f"❌ Invalid — {len(errs)} issue(s):\n" + "\n".join(f"- {e}" for e in errs)

                    reply = f"*via {backend_label}* · `{response.tool_name}`\n\n"
                    reply += summary if summary else f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"

                except Exception as e:
                    reply = f"❌ Error: {e}"

            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🎙️ IdemAI Core")
st.caption("Multi-language ASR data factory — Efik · Ibibio · Nigerian English")

tab_studio, tab_metrics, tab_scripts, tab_aziz = st.tabs(
    ["🎙️ Recording Studio", "📊 Metrics", "📝 Scripts", "🤖 Aziz"]
)

with tab_studio:
    st.components.v1.iframe(
        src="http://localhost:8001/studio",
        height=640,
        scrolling=True,
    )

with tab_metrics:
    render_metrics()

with tab_scripts:
    render_scripts()

with tab_aziz:
    render_aziz()

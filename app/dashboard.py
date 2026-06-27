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

import streamlit as st

# Resolve paths relative to repo root regardless of CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _REPO_ROOT / "master_manifest.jsonl"

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
    api_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required for the Aziz Agent console",
    )
    language_filter = st.selectbox(
        "Language filter (scripts view)",
        options=["yo", "efi", "ibb"],
        format_func=lambda c: {"yo": "Yoruba", "efi": "Efik", "ibb": "Ibibio"}[c],
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
        lang_names = {"yo": "Yoruba", "efi": "Efik", "ibb": "Ibibio"}
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
        st.warning("Set your Anthropic API key in the sidebar to use the Aziz Agent.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type a command… e.g. 'validate this Yoruba text: Ẹ káàárọ̀'")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Aziz is thinking…"):
                try:
                    from agent.aziz_orchestrator import AzizOrchestrator
                    aziz = AzizOrchestrator(
                        api_key=api_key,
                        config={"manifest_path": manifest_path},
                    )
                    response = aziz.run(user_input)
                    reply = (
                        f"**Intent:** `{response.intent}`\n\n"
                        f"```json\n{response.result}\n```"
                    )
                except Exception as e:
                    reply = f"❌ Error: {e}"

            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🎙️ IdemAI Core")
st.caption("Multi-language ASR data factory — Efik · Ibibio · Yoruba")

tab_metrics, tab_scripts, tab_aziz = st.tabs(["📊 Metrics", "📝 Scripts", "🤖 Aziz"])

with tab_metrics:
    render_metrics()

with tab_scripts:
    render_scripts()

with tab_aziz:
    render_aziz()

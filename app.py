"""
app.py
------
Streamlit demo UI. Shows every step of the agent's reasoning live --
this visibility is what sells "agentic" behavior in a viva, not just the
final answer.

Run with: streamlit run app.py
"""

import streamlit as st
from agent import ResearchAgent

st.set_page_config(page_title="Research Literature Agent", layout="wide")
st.title("Autonomous Research Literature Review Agent")
st.caption(
    "Decomposes your question -> searches arXiv -> synthesizes an answer -> "
    "checks itself -> refines once if the first pass was insufficient."
)

api_key = st.sidebar.text_input("Gemini API Key", type="password",
                                 help="Or set GEMINI_API_KEY env var and leave blank")

question = st.text_input(
    "Research question",
    placeholder="e.g. What are the main approaches to reducing hallucination in RAG systems?"
)

run_btn = st.button("Run agent")

if run_btn and question.strip():
    agent = ResearchAgent(api_key=api_key or None)

    with st.spinner("Running agent pipeline..."):
        try:
            result = agent.run(question)
        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.stop()

    st.subheader("Final Answer")
    st.write(result["answer"])

    st.subheader("Self-evaluation verdict")
    v = result["final_verdict"]
    st.json(v)
    st.caption(f"Refinement iterations used: {result['refinements_used']} (capped at 1)")

    st.subheader("Papers used")
    for p in result["papers"]:
        st.markdown(f"**{p['title']}**  \n{', '.join(p['authors'][:4])}  \n[{p['url']}]({p['url']})")

    with st.expander("Full agent reasoning trace (for viva demo)"):
        for step in result["trace"]:
            st.markdown(f"**{step['step']}**")
            st.json(step["data"])

elif run_btn:
    st.warning("Enter a question first.")

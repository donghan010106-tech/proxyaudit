"""
ProxyAudit — certificate-guided fairness console (Streamlit edition)

Reproduces, as a Streamlit app, the two standalone HTML demos used in the
paper "A Comparative Study of Early Fusion and Late Fusion for Fair
AI-Based Resume Screening Systems":
  1) auditor_demo.html       -> "Audit dashboard" tab
  2) recommender_demo.html   -> "Recovery feed & chat" tab

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Optional: to enable live LLM-phrased chat answers, set an Anthropic API key
either in a local .streamlit/secrets.toml file:
    ANTHROPIC_API_KEY = "sk-ant-..."
or as an environment variable ANTHROPIC_API_KEY. Without a key the chat
still works, using the built-in template fallback grounded in the audit data.
"""

import time
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import data

st.set_page_config(page_title="ProxyAudit Console", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def fmt_ci(mean, lo, hi, digits=3):
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def get_anthropic_client():
    """Return an Anthropic client if a key is configured, else None."""
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def local_answer(text: str) -> tuple[str, str]:
    """Template fallback answer, grounded only in the structured case data."""
    text_l = text.lower()
    for c in data.CASES:
        if c["id"] in text_l.split() or f"case {c['id']}" in text_l or f"case{c['id']}" in text_l:
            return c["message"], "local fallback · from the structured certificate"
    if any(k in text_l for k in ["gap", "parity", "overall", "summary"]):
        return (
            f"Across the audit the parity gap falls from {data.PARITY_BEFORE:.3f} to "
            f"{data.PARITY_AFTER:.3f} (−60%) once the localized channel is neutralized, "
            f"at an AUC cost of {data.AUC_COST:.3f}.",
            "local fallback",
        )
    return (
        "Ask about case 0, 9, 20 or 94, or about the overall parity gap.",
        "local fallback",
    )


def llm_answer(client, text: str) -> tuple[str, str]:
    ctx_lines = []
    for c in data.CASES:
        ctx_lines.append(
            f"case {c['id']} ({c['group']}, score effect {c['eff']}): verdict={c['error_type']}, "
            f"severity={c['severity']}, action={c['action_code']}. {c['message']}"
        )
    ctx = "\n".join(ctx_lines)
    prompt = (
        "You are a hiring-compliance assistant. Using ONLY the audit facts below, answer the "
        "question in 1-2 clear sentences. Do not invent numbers or names.\n\n"
        f"AUDIT FACTS:\n{ctx}\n\n"
        f"The channel that leaks the protected attribute is the {data.CHANNEL}. "
        f"Overall parity gap falls {data.PARITY_BEFORE} -> {data.PARITY_AFTER} after repair.\n\n"
        f"QUESTION: {text}"
    )
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if txt:
            return txt, "phrased by Claude"
    except Exception:
        pass
    return local_answer(text)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("ProxyAudit")
st.sidebar.caption(
    "Certificate-guided fairness console for resume screening.\n\n"
    "Real FairCVdb (gender). Numbers measured over 20 seeds (95% bootstrap). "
    "Research demo, not a deployed system."
)
tab_choice = st.sidebar.radio("View", ["Audit dashboard", "Recovery feed & chat"])

client = get_anthropic_client()
st.sidebar.divider()
if client:
    st.sidebar.success("Anthropic API key detected — chat answers will be LLM-phrased.")
else:
    st.sidebar.info("No Anthropic API key found — chat will use the built-in template fallback.")

# ----------------------------------------------------------------------
# Tab 1: Audit dashboard  (auditor_demo.html)
# ----------------------------------------------------------------------

if tab_choice == "Audit dashboard":
    st.title("ProxyAudit — certificate-guided fairness console")
    st.caption(
        "Real FairCVdb (gender). Numbers are measured over 20 seeds (95% bootstrap). "
        "Research demo, not a deployed system."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Repair strategies (measured)")
        rows = []
        for m in data.METHODS:
            rows.append({
                "Strategy": m["name"],
                "DP gap [95% CI]": fmt_ci(*m["dp"]),
                "fair-acc.": f"{m['acc']:.4f}",
                "cert.": "✓" if m["cert"] else "×",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f"Selective vs blanket retraining: Wilcoxon p={data.SIG['wilcoxon_p']:.1e}, "
            f"Cohen's d={data.SIG['cohens_d']:.1f}."
        )

    with col2:
        st.subheader("2. Fairness vs utility")
        fig = go.Figure()
        for m in data.METHODS:
            is_cert = m["cert"]
            is_unaware = "Unaware" in m["name"]
            color = "#2a9d8f" if is_cert else ("#d1495b" if is_unaware else "#9aa3b2")
            fig.add_trace(go.Scatter(
                x=[m["dp"][0]], y=[m["acc"]],
                mode="markers",
                marker=dict(size=18 if is_cert else 12,
                            symbol="star" if is_cert else "circle",
                            color=color),
                name=m["name"],
            ))
        fig.update_layout(
            xaxis=dict(title="DP gap (lower = fairer)", autorange="reversed"),
            yaxis=dict(title="fair-label accuracy"),
            legend=dict(font=dict(size=10)),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Residual-leakage probe (honest bound)")
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        st.metric("raw proxy block (non-linear AUC of A)", f"{data.LEAK['raw_block_auc'][0]:.2f}")
    with c2:
        st.metric("after linear orthogonalization", f"{data.LEAK['after_linear_orth_auc'][0]:.2f}")
    with c3:
        st.markdown(
            "A non-linear probe still recovers some attribute signal after a linear repair. "
            "We report this residual rather than assume it away; it is the concrete target for "
            "a non-linear neutralizer."
        )

    st.subheader("4. Per-decision certificate (illustrative)")
    st.caption(
        "For one applicant the audit issues a verdict from the directed counterfactual: "
        "necessity (does removing the proxy flip the decision?) and sufficiency (does the "
        "proxy signature alone reproduce it?)."
    )
    illustrative = [
        {"t": "Disadvantaged, rejected", "nec": True, "suf": True, "v": "caused → reversed", "c": "#d1495b"},
        {"t": "Disadvantaged, rejected", "nec": False, "suf": True, "v": "suppressed but not caused", "c": "#b08900"},
        {"t": "Advantaged, selected", "nec": True, "suf": True, "v": "inflated → reversed", "c": "#3a6ea5"},
    ]
    for k in illustrative:
        st.markdown(
            f"""
<div style="display:flex;gap:10px;align-items:center;margin:6px 0;padding:10px;
border:1px solid #e5e7eb;border-radius:9px;">
<span style="width:10px;height:10px;border-radius:50%;background:{k['c']};display:inline-block"></span>
<b>{k['t']}</b>
<span style="color:#6b7280;font-size:13px">necessity {'✓' if k['nec'] else '×'} · sufficiency {'✓' if k['suf'] else '×'}</span>
<span style="margin-left:auto;font-weight:700;color:{k['c']}">{k['v']}</span>
</div>
""",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------
# Tab 2: Recovery feed & chat  (recommender_demo.html)
# ----------------------------------------------------------------------

else:
    st.title("ProxyAudit — fairness audit & recovery console")
    st.caption(
        "Each hiring decision is localized to a proxy channel, certified as caused or not, "
        "and routed to a recovery action."
    )

    if "feed_idx" not in st.session_state:
        st.session_state.feed_idx = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    frac = 1 - (st.session_state.feed_idx / len(data.CASES)) * 0.60
    gap_now = data.PARITY_BEFORE * frac

    sev_colors = {"high": "#c8442b", "medium": "#2f6db0", "low": "#8a7320"}
    sev_bg = {"high": "#fbeae6", "medium": "#e8f0f9", "low": "#f6f0db"}

    def render_case_html(c):
        color = sev_colors[c["severity"]]
        bg = sev_bg[c["severity"]]
        steps_html = "".join(f"<li>{s}</li>" for s in c["steps"])
        return f"""
<div style="border:1px solid #e4e8ee;border-left:4px solid {color};border-radius:9px;
padding:10px 12px;margin:9px 0;background:#fff;">
<div style="font-size:11.5px;color:#6b7785;font-family:monospace">
case {c['id']} · {c['group']} · channel: {data.CHANNEL} · score effect {'+' if c['eff']>0 else ''}{c['eff']:.2f}
</div>
<div style="font-weight:700;font-size:14px;margin:3px 0">
{data.VLABEL[c['error_type']]}
<span style="font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:5px;
background:{bg};color:{color};margin-left:6px">{c['severity'].upper()}</span>
</div>
<div style="font-size:13px"><b>Recommendation:</b> {c['message']}</div>
<ul style="margin:7px 0 0;padding-left:18px;font-size:12px;color:#3a4452">{steps_html}</ul>
<div style="font-size:11px;color:#6b7785;margin-top:6px">action <b>{c['action_code']}</b> ·
target <b>{'case log' if c['action_code']=='LOG_AND_MONITOR' else 'application'}</b></div>
</div>
"""

    bcol1, bcol2, bcol3 = st.columns([2, 1, 1])
    with bcol2:
        run_clicked = st.button("▶ Run audit", use_container_width=True)
    with bcol3:
        if st.button("Reset", use_container_width=True):
            st.session_state.feed_idx = 0
            st.rerun()

    progress_box = bcol1.empty()
    progress_box.progress(
        min(max(frac, 0.0), 1.0),
        text=f"Group parity gap — before {data.PARITY_BEFORE:.3f}, now {gap_now:.3f}",
    )

    st.subheader("Step 5 — recovery recommendations")
    feed_box = st.container()

    if run_clicked:
        # Animate within this single script run: progressively reveal each
        # case and update the progress bar, without calling st.rerun().
        st.session_state.feed_idx = 0
        shown_html = ""
        with feed_box:
            placeholder = st.empty()
        for k, c in enumerate(data.CASES, start=1):
            st.session_state.feed_idx = k
            shown_html += render_case_html(c)
            placeholder.markdown(shown_html, unsafe_allow_html=True)
            new_frac = 1 - (k / len(data.CASES)) * 0.60
            progress_box.progress(
                min(max(new_frac, 0.0), 1.0),
                text=f"Group parity gap — before {data.PARITY_BEFORE:.3f}, "
                     f"now {data.PARITY_BEFORE * new_frac:.3f}",
            )
            time.sleep(0.5)
        st.success(
            f"Repair complete — parity gap reduced ~60% "
            f"(−{data.PARITY_BEFORE - data.PARITY_BEFORE * new_frac:.3f})."
        )
    else:
        with feed_box:
            if st.session_state.feed_idx == 0:
                st.info("Click **Run audit** to play back the per-decision recommendations.")
            else:
                shown_html = "".join(render_case_html(c) for c in data.CASES[: st.session_state.feed_idx])
                st.markdown(shown_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("Ask the auditor")
    st.caption(
        'Ask about any decision — e.g. "why was case 9 rejected?" or "what should we do about '
        'case 20?" (grounded in the audit above)'
    )

    for role, msg, src in st.session_state.chat_history:
        with st.chat_message("user" if role == "me" else "assistant"):
            st.markdown(msg)
            if src:
                st.caption(src)

    user_q = st.chat_input("Ask about a case…")
    if user_q:
        st.session_state.chat_history.append(("me", user_q, None))
        if client:
            answer, src = llm_answer(client, user_q)
        else:
            answer, src = local_answer(user_q)
        st.session_state.chat_history.append(("ai", answer, src))
        st.rerun()

"""ProxyAudit -- interactive fairness auditor (role-based, professional template).

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

A product-style auditor with two business roles that share one audit:

  * Recruiter / auditor  -- executive summary, fairness scorecard, bias
        localization, a live repair simulator, per-pool causal certificates,
        and a candidate-pool diff.
  * Job-seeker / applicant -- a personal "fairness report card": were you
        suppressed or inflated by a proxy? Is the proxy channel necessary,
        sufficient, or a *cause* of your decision? What would a fair model do?

Everything is computed by the `proxyaudit` package; nothing is mocked.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from proxyaudit.data import make_faircv_recipe, make_fairjob_sim, COMPETENCY, COMP_NAMES
from proxyaudit.pipeline import run_pcln
from proxyaudit.counterfactual import individual_certificate

st.set_page_config(page_title="ProxyAudit", layout="wide", page_icon="\u2696\ufe0f",
                   initial_sidebar_state="expanded")

# ---- palette ------------------------------------------------------------- #
INK, PROXY, COMP, GOOD, WARN, BAD = "#1b2330", "#E4572E", "#2E86AB", "#2E9E5B", "#E0A100", "#C2384A"

st.markdown(f"""
<style>
:root {{ --ink:{INK}; --proxy:{PROXY}; --comp:{COMP}; --good:{GOOD}; --bad:{BAD}; }}
.block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
.hero {{ background: linear-gradient(110deg,#101826 0%,#1f3550 60%,#2E86AB 130%);
        color:#fff; padding:22px 26px; border-radius:16px; margin-bottom:14px;
        box-shadow:0 8px 26px rgba(16,24,38,.20); }}
.hero h1 {{ margin:0; font-size:1.55rem; letter-spacing:.2px; }}
.hero p  {{ margin:.35rem 0 0; opacity:.92; font-size:.96rem; }}
.card {{ background:#fff; border:1px solid #e7ebf0; border-radius:14px;
        padding:16px 18px; box-shadow:0 2px 10px rgba(20,30,50,.05); height:100%; }}
.card h4 {{ margin:.1rem 0 .5rem; color:var(--ink); font-size:1rem; }}
.kpi {{ font-size:1.9rem; font-weight:700; line-height:1.1; }}
.kpi small {{ font-size:.8rem; font-weight:600; }}
.badge {{ display:inline-block; padding:4px 11px; border-radius:999px;
         font-size:.8rem; font-weight:700; color:#fff; margin:2px 4px 2px 0; }}
.b-good {{ background:var(--good); }} .b-bad {{ background:var(--bad); }}
.b-warn {{ background:{WARN}; }} .b-mut {{ background:#7a869a; }}
.b-proxy {{ background:var(--proxy); }}
.meter {{ height:16px; border-radius:9px; background:#eef1f5; position:relative; overflow:hidden; }}
.meter > span {{ position:absolute; top:0; bottom:0; }}
.verdict {{ border-radius:13px; padding:16px 18px; font-size:1.02rem; font-weight:600; }}
.v-cause {{ background:#fdece8; border:1px solid #f3b3a4; color:#8f2d18; }}
.v-part  {{ background:#fff6e0; border:1px solid #f0d493; color:#7a5800; }}
.v-clear {{ background:#e7f6ec; border:1px solid #a9dcb9; color:#1d6b39; }}
.small {{ color:#5a6678; font-size:.86rem; }}
.step {{ border-left:3px solid var(--comp); padding:6px 0 6px 12px; margin:6px 0; }}
hr {{ margin:.7rem 0; }}
</style>
""", unsafe_allow_html=True)


# ---- audit (cached) ------------------------------------------------------ #
@st.cache_data(show_spinner="Auditing the candidate pool\u2026")
def run_audit(testbed, kind, mode, seed, rho, n_sem):
    if testbed.startswith("FairCV"):
        ds = make_faircv_recipe(n=8000, seed=seed, proxy_strength=rho,
                                n_proxy=(4 if n_sem else 8), n_semantic=n_sem)
    else:
        ds = make_fairjob_sim(n=20000, seed=seed)
    out = run_pcln(ds, kind=kind, seed=seed, neutralize_mode=mode,
                   n_explain=400, verbose=False)
    return ds, out


def meter(frac, color, label=""):
    frac = max(0.0, min(1.0, float(frac)))
    return (f'<div class="meter"><span style="left:0;width:{frac*100:.1f}%;'
            f'background:{color};"></span></div>'
            f'<div class="small">{label}</div>')


def badge(text, cls):
    return f'<span class="badge {cls}">{text}</span>'


# ---- sidebar ------------------------------------------------------------- #
st.sidebar.markdown("### \u2696\ufe0f ProxyAudit")
st.sidebar.caption("Causal certificates for proxy discrimination")
role = st.sidebar.radio("Open the console as", ["\U0001F3E2 Recruiter / auditor",
                                                 "\U0001F9D1 Job-seeker / applicant"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Audit configuration**")
testbed = st.sidebar.selectbox("Scenario", ["FairCVtest recipe (recruitment)", "FairJob twin (ads)"])
kind = st.sidebar.selectbox("Model under audit", ["lr", "hgb"],
                            format_func=lambda k: {"lr": "Logistic regression", "hgb": "Gradient boosting"}[k])
mode = st.sidebar.selectbox("Repair operator", ["orthogonalize", "drop", "suppress"],
                            help="Orthogonalize removes only the protected direction "
                                 "inside the proxy block, preserving legitimate signal.")
seed = st.sidebar.slider("Random seed", 0, 9, 0)
is_cv = testbed.startswith("FairCV")
rho = st.sidebar.slider("Proxy strength \u03c1", 0.30, 0.95, 0.85, 0.05,
                        help="How strongly the hidden channel encodes the protected attribute.") if is_cv else 0.6
n_sem = st.sidebar.slider("Mixed (semantic) proxies", 0, 4, 0,
                          help="Proxies that ALSO carry legitimate signal \u2014 where "
                               "orthogonalize beats deletion and causation turns conservative.") if is_cv else 0
st.sidebar.markdown("---")
st.sidebar.caption("FairJob-twin numbers are a protocol demo, not a real measurement.")

ds, out = run_audit(testbed, kind, mode, seed, rho, n_sem)
arr = out["_arrays"]; pcc = out["pcc"]; A = np.asarray(arr["Ate"])
names = arr["feature_names"]; Xte = arr["Xte"].reset_index(drop=True)
proxies = set(out["localized_proxies"])
b, a = out["before_fair"], out["after_fair"]
bp, ap = out["before_perf"], out["after_perf"]


# ========================================================================== #
#  RECRUITER                                                                  #
# ========================================================================== #
if role.startswith("\U0001F3E2"):
    st.markdown('<div class="hero"><h1>Recruiter \u00b7 candidate-pool fairness audit</h1>'
                '<p>Localize the hidden bias channel, certify which decisions it '
                '<b>causes</b>, simulate a surgical repair, and ship a defensible report.</p></div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["\U0001F4CA Executive summary", "\u2696\ufe0f Fairness scorecard",
                    "\U0001F50D Bias localization", "\U0001F527 Repair simulator",
                    "\U0001F9FE Causal certificates", "\U0001F465 Candidate pool"])

    # --- Executive summary ---
    with tabs[0]:
        dpred = 100 * (b["DP_gap"] - a["DP_gap"]) / max(b["DP_gap"], 1e-9)
        c = st.columns(4)
        c[0].markdown(f'<div class="card"><h4>Disparity removed</h4>'
                      f'<div class="kpi" style="color:{GOOD}">{dpred:.0f}%</div>'
                      f'<div class="small">DP gap {b["DP_gap"]:.3f} \u2192 {a["DP_gap"]:.3f}</div></div>', unsafe_allow_html=True)
        c[1].markdown(f'<div class="card"><h4>Four-fifths rule</h4>'
                      f'<div class="kpi" style="color:{GOOD if a["DI"]>=0.8 else BAD}">{a["DI"]:.2f}</div>'
                      f'<div class="small">disparate impact (\u2265 0.80 passes)</div></div>', unsafe_allow_html=True)
        c[2].markdown(f'<div class="card"><h4>Utility cost</h4>'
                      f'<div class="kpi">{ap["AUC"]-bp["AUC"]:+.3f}</div>'
                      f'<div class="small">AUC {bp["AUC"]:.3f} \u2192 {ap["AUC"]:.3f}</div></div>', unsafe_allow_html=True)
        c[3].markdown(f'<div class="card"><h4>Proxy-caused rejections</h4>'
                      f'<div class="kpi" style="color:{PROXY}">{pcc["caused_A1"]:.0%}</div>'
                      f'<div class="small">{pcc["n_caused_A1"]}/{pcc["n_adv_rejected"]} disadvantaged</div></div>', unsafe_allow_html=True)
        st.markdown("")
        if a["DI"] >= 0.8 > b["DI"]:
            st.success("**Pass.** After the targeted repair the pool clears the four-fifths rule "
                       f"and {pcc['caused_A1']:.0%} of disadvantaged rejections that were *caused* "
                       "by the proxy channel are removed.")
        if a["EOO_gap"] > b["EOO_gap"] + 0.02:
            st.warning("**Trade-off to disclose.** Minimizing demographic parity raised the "
                       f"equal-opportunity gap ({b['EOO_gap']:.3f} \u2192 {a['EOO_gap']:.3f}). "
                       "If equal true-positive rates is your legal standard, target EOO directly.")
        st.markdown('<div class="card"><h4>What this audit asserts</h4>'
                    '<div class="step">A hidden <b>proxy channel</b> was localized from the model\u2019s own explanations.</div>'
                    '<div class="step">Its exact contribution to the group disparity was measured (Proposition\u00a01).</div>'
                    '<div class="step">For each rejected applicant we tested whether the channel was '
                    '<b>necessary</b> and <b>sufficient</b> \u2014 a per-decision <b>cause</b>.</div>'
                    '<div class="step">A surgical repair removed that channel while preserving legitimate signal.</div>'
                    '</div>', unsafe_allow_html=True)

    # --- Fairness scorecard ---
    with tabs[1]:
        st.subheader("Before \u2192 after the targeted repair")
        rows = [("Demographic parity gap", b["DP_gap"], a["DP_gap"], "lower"),
                ("Disparate impact (DI)", b["DI"], a["DI"], "higher"),
                ("Equal-opportunity gap", b["EOO_gap"], a["EOO_gap"], "lower"),
                ("AUC (utility)", bp["AUC"], ap["AUC"], "higher"),
                ("Calibration error (ECE)", out["before_trust"]["ECE"], out["after_trust"]["ECE"], "lower")]
        for nm, bv, av, better in rows:
            improved = (av < bv) if better == "lower" else (av > bv)
            col = GOOD if improved else BAD
            cc = st.columns([2.2, 1, 1, 2.4])
            cc[0].markdown(f"**{nm}**")
            cc[1].markdown(f"<span class='small'>before</span><br><b>{bv:.3f}</b>", unsafe_allow_html=True)
            cc[2].markdown(f"<span class='small'>after</span><br><b style='color:{col}'>{av:.3f}</b>", unsafe_allow_html=True)
            frac = av if better == "higher" else max(0, 1 - av / max(bv, 1e-9))
            cc[3].markdown(meter(frac, col, "better \u2192" if improved else "worse"), unsafe_allow_html=True)
        st.caption("Fairness \u00b7 Faithfulness \u00b7 Trust are scored together so a fairness gain "
                   "cannot silently degrade calibration or explanation quality.")

    # --- Bias localization ---
    with tabs[2]:
        st.subheader("Where is the bias? \u2014 the proxy channel, ranked")
        cL, cR = st.columns([1.2, 1.0])
        with cL:
            tbl = pd.DataFrame(out["pls_table"]).head(14)[["feature", "PLS", "reconstructability", "dp_share"]]
            st.dataframe(tbl.style.apply(
                lambda r: ["background-color:#fdecec" if r["feature"] in proxies else "" for _ in r], axis=1)
                .format({"PLS": "{:.3f}", "reconstructability": "{:.3f}", "dp_share": "{:.3f}"}),
                use_container_width=True, height=420)
        with cR:
            pls = arr["pls"]; order = np.argsort(-np.abs(pls["delta"]))[:12]
            fig, ax = plt.subplots(figsize=(4.7, 4.4))
            nm = [pls["names"][i] for i in order]; vv = [pls["delta"][i] for i in order]
            ax.barh(range(len(nm)), vv, color=[PROXY if n in proxies else COMP for n in nm])
            ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=8); ax.invert_yaxis()
            ax.axvline(0, color="gray", lw=.8); ax.set_xlabel(r"$\Delta_j$ (exact DP contribution)")
            ax.set_title("Per-feature parity decomposition", fontsize=10)
            for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
            st.pyplot(fig)
            st.markdown(f"$\\sum_j\\Delta_j$ = **{pls['dp_score_gap']:.3f}** "
                        "(equals the score-level parity gap, exactly).")
        st.info("A feature is flagged only when it **both** reconstructs the protected "
                "attribute **and** drives the gap \u2014 the Proxy Leakage Score "
                "$\\mathrm{PLS}=\\sqrt{R\\cdot D}$.")

    # --- Repair simulator ---
    with tabs[3]:
        st.subheader("Repair simulator \u2014 fairness vs. utility")
        st.caption("Change the **repair operator** in the sidebar to compare. "
                   "On mixed proxies, deletion sacrifices utility that orthogonalization keeps.")
        c = st.columns(3)
        c[0].metric("DP gap after", f"{a['DP_gap']:.3f}", f"{a['DP_gap']-b['DP_gap']:+.3f}", delta_color="inverse")
        c[1].metric("AUC after", f"{ap['AUC']:.3f}", f"{ap['AUC']-bp['AUC']:+.3f}")
        c[2].metric("Decision stability", f"{out['after_trust']['decision_stability']:.0%}",
                    help="Share of decisions unchanged by the repair.")
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        ax.scatter(b["DP_gap"], bp["AUC"], s=150, color=PROXY, edgecolor="white", zorder=3, label="before (unaware)")
        ax.scatter(a["DP_gap"], ap["AUC"], s=190, marker="*", color=GOOD, edgecolor="white", zorder=3, label=f"after ({mode})")
        ax.annotate("before", (b["DP_gap"], bp["AUC"]), xytext=(6, -10), textcoords="offset points", fontsize=8)
        ax.annotate("after", (a["DP_gap"], ap["AUC"]), xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.set_xlabel("DP gap (lower = fairer)"); ax.set_ylabel("AUC (higher = better)")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.2)
        st.pyplot(fig)

    # --- Causal certificates ---
    with tabs[4]:
        st.subheader("Per-pool causal certificate (disadvantaged rejections)")
        c = st.columns(3)
        for col, (lab, val, desc) in zip(c, [
            ("Necessary", pcc["necessity_A1"], "removing the proxy flips the decision"),
            ("Sufficient", pcc["sufficiency_A1"], "the proxy signature alone reproduces it"),
            ("CAUSED", pcc["caused_A1"], "necessary AND sufficient")]):
            color = PROXY if lab == "CAUSED" else COMP
            col.markdown(f'<div class="card"><h4>{lab}</h4>'
                         f'<div class="kpi" style="color:{color}">{val:.1%}</div>'
                         f'<div class="small">{desc}</div>{meter(val, color)}</div>', unsafe_allow_html=True)
        st.markdown("")
        st.info(f"On this pool **{pcc['n_caused_A1']} of {pcc['n_adv_rejected']}** disadvantaged "
                "rejections are *caused* by the proxy channel. Note how **sufficiency** falls when "
                "you add *mixed* proxies in the sidebar \u2014 the certificate refuses to call a "
                "channel a pure cause when it also does legitimate work.")
        # group effect distribution
        a_ = pcc["_arrays"]; eff = np.asarray(a_["effect"]); At = np.asarray(a_["Ate"])
        fig, ax = plt.subplots(figsize=(8.2, 3.0))
        ax.hist(eff[At == 0], bins=45, alpha=.6, color=COMP, density=True, label="A=0 advantaged")
        ax.hist(eff[At == 1], bins=45, alpha=.6, color="#E8567A", density=True, label="A=1 disadvantaged")
        ax.axvline(0, color="gray", ls="--", lw=.9); ax.set_xlabel("cluster-CF effect e(x)")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8); st.pyplot(fig)

    # --- Candidate pool ---
    with tabs[5]:
        st.subheader("Who is selected \u2014 before vs after")
        yp0, yp1 = np.asarray(arr["yp_un"]), np.asarray(arr["yp_pc"])
        rows = []
        for g, lab in [(1, "A=1 (disadvantaged)"), (0, "A=0 (advantaged)")]:
            m = A == g
            rows.append({"group": lab, "selected before": f"{100*yp0[m].mean():.1f}%",
                         "selected after": f"{100*yp1[m].mean():.1f}%",
                         "newly selected": int(np.sum(yp1[m] > yp0[m])),
                         "newly dropped": int(np.sum(yp1[m] < yp0[m]))})
        st.table(pd.DataFrame(rows))
        st.caption("Equal-rate selection budget held fixed before and after, so the change "
                   "reflects re-allocation, not a different number of hires.")


# ========================================================================== #
#  JOB-SEEKER                                                                 #
# ========================================================================== #
else:
    st.markdown('<div class="hero"><h1>Your fairness report card</h1>'
                '<p>Was your decision driven by your competencies \u2014 or by a hidden '
                'proxy for a protected attribute the model was never supposed to use?</p></div>',
                unsafe_allow_html=True)

    a_ = pcc["_arrays"]
    d0 = np.asarray(a_["d0"]); d1 = np.asarray(a_["d1"]); dsuf = np.asarray(a_["d_suf"])
    caused_idx = np.where((A == 1) & (d0 == 0) & (d1 == 1) & (dsuf == 0))[0]
    default_i = int(caused_idx[0]) if len(caused_idx) else 0

    cc = st.columns([1.4, 1])
    with cc[0]:
        choice = st.radio("Choose an application to inspect", horizontal=True,
                          options=["A flagged case (proxy-caused)", "Pick by index"])
    with cc[1]:
        if choice.startswith("A flagged") and len(caused_idx):
            i = default_i
            st.caption(f"Showing a representative proxy-caused rejection (#{i}).")
        else:
            i = int(st.number_input("Applicant index", 0, len(Xte) - 1, default_i, 1))
    i = int(i)
    cert = individual_certificate(pcc, i)
    grp = "disadvantaged group (A=1)" if cert["group"] == 1 else "advantaged group (A=0)"

    L, R = st.columns([1.05, 1])
    with L:
        st.markdown('<div class="card"><h4>\U0001F4C4 Your application</h4>', unsafe_allow_html=True)
        prof = {COMP_NAMES[COMPETENCY.index(c)]: round(float(Xte[c].iloc[i]), 2)
                for c in names if c in COMPETENCY}
        st.dataframe(pd.DataFrame([prof]).T.rename(columns={0: "score"}), use_container_width=True, height=240)
        st.markdown(f'<div class="small">Protected group: <b>{grp}</b> \u2014 never shown to the '
                    'model, but it can leak in through correlated features.</div></div>', unsafe_allow_html=True)
    with R:
        dec_ok = cert["factual_decision"] == 1
        st.markdown(f'<div class="card"><h4>\U0001F4DD The decision</h4>'
                    f'<div class="kpi" style="color:{GOOD if dec_ok else BAD}">'
                    f'{"SELECTED" if dec_ok else "NOT selected"}</div>'
                    f'<div class="small">at the current hiring budget</div><hr>', unsafe_allow_html=True)
        eff = cert["cluster_effect"]; mag = min(1.0, abs(eff) / 3.0)
        clr = BAD if eff < 0 else GOOD
        st.markdown(f'<b>Hidden-channel effect on your score:</b> '
                    f'<span style="color:{clr};font-weight:700">{eff:+.2f}</span> '
                    f'({"suppressed you" if eff<0 else "boosted you"})'
                    f'{meter(mag, clr)}</div>', unsafe_allow_html=True)

    # verdict
    nec, suf = cert["necessary"], cert["sufficient"]
    st.markdown("### Verdict")
    cb = st.columns(3)
    cb[0].markdown(badge(f"Necessary: {'YES' if nec else 'no'}", "b-bad" if nec else "b-mut"), unsafe_allow_html=True)
    cb[1].markdown(badge(f"Sufficient: {'YES' if suf else 'no'}", "b-bad" if suf else "b-mut"), unsafe_allow_html=True)
    cb[2].markdown(badge("CAUSE" if (nec and suf) else "not a cause", "b-bad" if (nec and suf) else "b-good"), unsafe_allow_html=True)

    if nec and suf:
        if cert["group"] == 1:
            st.markdown('<div class="verdict v-cause">\u26A0\uFE0F The hidden proxy channel is a '
                        '<b>cause</b> of your rejection. Removing it flips your decision, '
                        '<i>and</i> the proxy signature alone is enough to produce it. '
                        'Under the repaired (fair) model you would be <b>selected</b>.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict v-part">The hidden proxy channel is a <b>cause</b> of '
                        'your selection \u2014 an unfair advantage the repaired model would not grant.</div>',
                        unsafe_allow_html=True)
    elif nec:
        st.markdown('<div class="verdict v-part">The proxy channel was <b>necessary but not '
                    'sufficient</b> \u2014 it tipped the balance together with other factors.</div>',
                    unsafe_allow_html=True)
    elif suf:
        st.markdown('<div class="verdict v-part">The proxy channel <b>could</b> drive this decision, '
                    'but removing it does not change your outcome \u2014 other factors already decide it.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict v-clear">\u2705 Your decision is <b>not attributable</b> to '
                    'the proxy channel \u2014 it rests on your legitimate competencies.</div>',
                    unsafe_allow_html=True)

    st.markdown("")
    cf = st.columns(2)
    fair_ok = cert["decision_if_cluster_neutralized"] == 1
    cf[0].markdown(f'<div class="card"><h4>\U0001F501 What a fair model decides</h4>'
                   f'<div class="kpi" style="color:{GOOD if fair_ok else BAD}">'
                   f'{"SELECTED" if fair_ok else "NOT selected"}</div>'
                   f'<div class="small">with the proxy channel neutralized</div></div>', unsafe_allow_html=True)
    alone_ok = cert["decision_from_cluster_alone"] == 1
    cf[1].markdown(f'<div class="card"><h4>\U0001F9EA The proxy signature alone</h4>'
                   f'<div class="kpi" style="color:{BAD if not alone_ok else GOOD}">'
                   f'{"SELECTED" if alone_ok else "NOT selected"}</div>'
                   f'<div class="small">your proxy values on an otherwise-average profile</div></div>',
                   unsafe_allow_html=True)

    with st.expander("How this report card is computed", expanded=False):
        st.markdown(
            "- **Hidden-channel effect** $e(x)=f(x)-f(x^{\\mathrm{CF}})$: your score minus your "
            "score with the proxy block\u2019s protected direction removed.\n"
            "- **Necessary**: neutralizing the proxy cluster (holding everything else fixed) "
            "flips your decision.\n"
            "- **Sufficient**: injecting your proxy values onto an otherwise-average profile "
            "reproduces your decision.\n"
            "- **Cause** = necessary **and** sufficient \u2014 a per-decision certificate that the "
            "proxy channel, not your competencies, drove the outcome.\n\n"
            "These per-applicant certificates provably average back to the pool-level "
            "demographic-parity gap (the consistency theorem), so your individual report and "
            "the organization\u2019s audit are the same quantity at two resolutions.")

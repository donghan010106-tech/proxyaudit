"""
Measured numbers extracted from the original HTML demos (auditor_demo.html,
recommender_demo.html, demo_cases.json), kept verbatim so the Streamlit app
reproduces the same figures reported in the paper / console.
"""

CHANNEL = "demographic embedding (photo/name)"

# --- Section 1: repair strategies (Table 5 in the paper) ---
METHODS = [
    {"name": "Unaware (no repair)", "dp": [0.19234862001136693, 0.1727339493690332, 0.21471878756981005],
     "acc": 0.7723679369087362, "cert": False},
    {"name": "Group thresholds", "dp": [0.00028401630673668965, 4.395767761020131e-06, 0.0007804879713590951],
     "acc": 0.778847248440248, "cert": False},
    {"name": "Reweighing", "dp": [0.08730616244955544, 0.06341916682546084, 0.10788181167387298],
     "acc": 0.7859263179651611, "cert": False},
    {"name": "Eq.-odds post-processing", "dp": [0.01504915484064893, 4.67317457460259e-05, 0.030790204161131447],
     "acc": 0.778966756064032, "cert": False},
    {"name": "Blanket repair (retrain)", "dp": [0.07443555789832257, 0.0547894946992165, 0.09591804374921911],
     "acc": 0.7833750961895489, "cert": False},
    {"name": "Selective (ours, certified)", "dp": [0.04342031146826864, 0.011626880627271166, 0.0635028906816582],
     "acc": 0.7787027963257869, "cert": True},
]

SIG = {
    "selective_mean": 0.04342031146826864,
    "blanket_mean": 0.07443555789832257,
    "wilcoxon_p": 1.9073486328125e-06,
    "paired_t_p": 2.2511263713487059e-10,
    "cohens_d": 2.7759764835032086,
}

# --- Section 3: residual-leakage probe ---
LEAK = {
    "raw_block_auc": [0.896176568376589, 0.8890004645886836, 0.9059513241795338],
    "after_linear_orth_auc": [0.6637674286087503, 0.6503782218158864, 0.6834466838956532],
}

N_SEEDS = 20

# --- Section 4 / recovery feed: per-decision certificates ---
# Merged from recommender_demo.html (has score effect + group) and demo_cases.json
# (has the structured recovery action / steps).
CASES = [
    {
        "id": "9", "group": "disadvantaged", "eff": -1.09,
        "error_type": "proxy_caused_rejection", "severity": "high",
        "necessity": True, "sufficiency": True, "verdict": "caused → reversed",
        "action_code": "REVERSE_AND_REVIEW", "target": "application",
        "steps": [
            "Re-score the application with the named channel neutralized.",
            "Route to human review with the certificate attached.",
            "Notify the candidate that the decision is being re-assessed.",
        ],
        "message": (
            f"This rejection was caused by the {CHANNEL}: removing only that channel flips the "
            "decision, and the candidate clears the bar on competencies alone. Re-evaluate with "
            "the channel neutralized, route to human review, and notify the candidate of "
            "re-assessment."
        ),
    },
    {
        "id": "94", "group": "disadvantaged", "eff": -1.93,
        "error_type": "proxy_caused_rejection", "severity": "high",
        "necessity": True, "sufficiency": True, "verdict": "caused → reversed",
        "action_code": "REVERSE_AND_REVIEW", "target": "application",
        "steps": [
            "Re-score the application with the named channel neutralized.",
            "Route to human review with the certificate attached.",
            "Notify the candidate that the decision is being re-assessed.",
        ],
        "message": (
            f"This rejection was caused by the {CHANNEL}: removing only that channel flips the "
            "decision, and the candidate clears the bar on competencies alone. Re-evaluate with "
            "the channel neutralized, route to human review, and notify the candidate of "
            "re-assessment."
        ),
    },
    {
        "id": "0", "group": "disadvantaged", "eff": -2.01,
        "error_type": "proxy_suppressed_no_flip", "severity": "low",
        "necessity": False, "sufficiency": True, "verdict": "suppressed but not caused",
        "action_code": "LOG_AND_MONITOR", "target": "case_log",
        "steps": [
            "Record that the channel suppressed the score without changing the outcome.",
            "Add to the monitoring set in case the margin shifts.",
        ],
        "message": (
            f"The {CHANNEL} lowered this score but did not change the outcome; the rest of the "
            "profile already determined it. No reversal is required, but log the case and "
            "monitor the margin."
        ),
    },
    {
        "id": "20", "group": "advantaged", "eff": 2.33,
        "error_type": "proxy_inflated_selection", "severity": "medium",
        "necessity": True, "sufficiency": True, "verdict": "inflated → reversed",
        "action_code": "RECHECK_SELECTION", "target": "application",
        "steps": [
            "Re-score with the named channel neutralized.",
            "Confirm the candidate still clears the bar on competencies alone.",
        ],
        "message": (
            f"This selection was inflated by the {CHANNEL}. Re-score with the channel "
            "neutralized and confirm the candidate still qualifies on competencies alone "
            "before finalizing."
        ),
    },
]

VLABEL = {
    "proxy_caused_rejection": "Rejected — proxy-caused",
    "proxy_suppressed_no_flip": "Rejected — not proxy-caused",
    "proxy_inflated_selection": "Selected — proxy-inflated",
}

PARITY_BEFORE = 0.199
PARITY_AFTER = 0.079
AUC_COST = 0.014

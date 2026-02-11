# app.py
# Streamlit CS2 PrizePicks Props Model (Kills + Headshots)
# Run: streamlit run app.py

import os
import math
from datetime import datetime, date
import pandas as pd
import streamlit as st

APP_TITLE = "CS2 PrizePicks Props Model — Kills + Headshots"
DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")

# ----------------------------
# PrizePicks defaults
# ----------------------------
DEFAULT_THRESHOLDS = {
    "Bo1": {"kills": 2.5, "hs": 2.0},
    "Bo3": {"kills": 3.5, "hs": 3.0},
    "Bo5": {"kills": 4.5, "hs": 4.0},
}

BLOWOUT_ADJ = {
    "Low": 0.00,
    "Medium": -0.03,  # fewer rounds risk
    "High": -0.06,
}

COMPETITIVE_ADJ = {
    "Low": -0.03,     # likely stomp -> fewer rounds
    "Medium": 0.00,
    "High": +0.05,    # closer series -> more rounds
}

ROLE_CERTAINTY_THRESHOLD_BUMP = {
    "Confirmed": 0.0,
    "Uncertain": 1.0,  # widen thresholds if role unclear
}

# ----------------------------
# Utilities
# ----------------------------
def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        df = pd.DataFrame(columns=[
            "timestamp",
            "date",
            "match",
            "player",
            "format",
            "expected_maps",
            "kpm",
            "hs_pct",

            # core adjustments
            "role_adj",
            "opp_adj",
            "map_adj",

            # PrizePicks-specific context
            "blowout_risk",
            "competitiveness",
            "role_certainty",
            "blowout_adj",
            "competitive_adj",
            "threshold_bump",

            "total_adj",
            "kills_line",
            "hs_line",

            # thresholds used
            "kills_threshold_used",
            "hs_threshold_used",

            # outputs
            "base_kills",
            "adj_kills",
            "proj_headshots",
            "kill_edge",
            "hs_edge",
            "kills_pick",
            "hs_pick",
            "kills_confidence",
            "hs_confidence",

            # meta
            "notes",

            # results tracking
            "actual_kills",
            "actual_headshots",
            "kills_result",
            "hs_result",
        ])
        df.to_csv(HISTORY_PATH, index=False)

def load_history() -> pd.DataFrame:
    ensure_storage()
    try:
        return pd.read_csv(HISTORY_PATH)
    except Exception:
        return pd.DataFrame()

def save_row(row: dict):
    ensure_storage()
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def fmt_num(x: float, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"

def pick_from_edge(edge: float, threshold: float) -> str:
    if edge >= threshold:
        return "MORE"
    if edge <= -threshold:
        return "LESS"
    return "PASS"

def confidence_from_edge(edge: float, threshold: float) -> str:
    # PrizePicks-friendly: only push high/medium
    abs_edge = abs(edge)
    if abs_edge < threshold:
        return "Low"
    if abs_edge >= threshold + 2.0:
        return "High"
    return "Medium"

def compute_results(pick: str, actual: float, line: float) -> str:
    if pick not in ("MORE", "LESS"):
        return "NA"
    if pd.isna(actual) or pd.isna(line):
        return "NA"
    if actual == line:
        return "PUSH"
    if pick == "MORE":
        return "WIN" if actual > line else "LOSS"
    return "WIN" if actual < line else "LOSS"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

history = load_history()
tabs = st.tabs(["Model (PrizePicks)", "History", "Settings / Help"])

# ----------------------------
# Tab: Model
# ----------------------------
with tabs[0]:
    st.subheader("PrizePicks mode: wider thresholds + context + confidence")

    with st.sidebar:
        st.header("PrizePicks Thresholds")
        auto_thresholds = st.toggle("Auto thresholds by format", value=True)

        st.caption("Defaults: Bo1 (Kills 2.5 / HS 2.0), Bo3 (3.5 / 3.0), Bo5 (4.5 / 4.0)")
        manual_kill_threshold = st.number_input("Manual kills threshold", value=3.5, step=0.5)
        manual_hs_threshold = st.number_input("Manual headshots threshold", value=3.0, step=0.5)

        st.divider()
        st.header("Quick Defaults")
        default_bo3_maps = st.number_input("Default expected maps for Bo3", value=2.0, step=0.5)
        default_bo1_maps = st.number_input("Default expected maps for Bo1", value=1.0, step=0.5)
        default_bo5_maps = st.number_input("Default expected maps for Bo5", value=3.0, step=0.5)

        st.divider()
        st.header("Correlation Warning")
        st.caption("Avoid stacking: same player Kills + Headshots (high correlation). Prefer different matches.")

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown("### Inputs")
        d = st.date_input("Date", value=date.today())
        match = st.text_input("Match (optional)", placeholder="Team A vs Team B")
        player = st.text_input("Player", placeholder="Player name / handle")
        fmt = st.selectbox("Series format", ["Bo1", "Bo3", "Bo5"], index=1)

        suggested_maps = {"Bo1": default_bo1_maps, "Bo3": default_bo3_maps, "Bo5": default_bo5_maps}[fmt]
        expected_maps = st.number_input("Expected maps", value=float(suggested_maps), step=0.5, min_value=1.0)

        st.markdown("### Stats (pull manually)")
        kpm = st.number_input("HLTV Kills per Map (KPM) — recent window", value=18.0, step=0.1, min_value=0.0)
        hs_pct = st.number_input("Headshot % (decimal) — ex: 0.47", value=0.47, step=0.01, min_value=0.0, max_value=1.0)

        st.markdown("### Core adjustments (decimals)")
        role_adj = st.number_input("Role adj (entry +, passive -) ex: 0.05", value=0.00, step=0.01, min_value=-0.50, max_value=0.50)
        opp_adj = st.number_input("Opponent adj (tough -, weak +) ex: -0.05", value=0.00, step=0.01, min_value=-0.50, max_value=0.50)
        map_adj = st.number_input("Map pool adj (good +, bad -) ex: 0.05", value=0.00, step=0.01, min_value=-0.50, max_value=0.50)

        st.markdown("### PrizePicks context (round volume control)")
        blowout_risk = st.selectbox("Blowout risk", ["Low", "Medium", "High"], index=1)
        competitiveness = st.selectbox("Series competitiveness", ["Low", "Medium", "High"], index=1)
        role_certainty = st.selectbox("Role certainty", ["Confirmed", "Uncertain"], index=0)

        st.markdown("### PrizePicks lines")
        kills_line = st.number_input("Kills line", value=35.5, step=0.5, min_value=0.0)
        hs_line = st.number_input("Headshots line", value=16.5, step=0.5, min_value=0.0)

        notes = st.text_area("Notes (optional)", placeholder="Any context: map veto, role change, stand-in, etc.", height=100)

    # Thresholds
    base_kills_thr = DEFAULT_THRESHOLDS[fmt]["kills"]
    base_hs_thr = DEFAULT_THRESHOLDS[fmt]["hs"]
    bump = ROLE_CERTAINTY_THRESHOLD_BUMP[role_certainty]

    kills_threshold_used = (base_kills_thr + bump) if auto_thresholds else manual_kill_threshold
    hs_threshold_used = (base_hs_thr + bump) if auto_thresholds else manual_hs_threshold

    # PrizePicks contextual adjustments (round volume)
    blowout_adj = BLOWOUT_ADJ[blowout_risk]
    competitive_adj = COMPETITIVE_ADJ[competitiveness]

    # Total adjustment
    total_adj = role_adj + opp_adj + map_adj + blowout_adj + competitive_adj

    # Projections
    base_kills = kpm * expected_maps
    adj_kills = base_kills * (1.0 + total_adj)
    proj_headshots = adj_kills * hs_pct

    # Edges
    kill_edge = adj_kills - kills_line
    hs_edge = proj_headshots - hs_line

    # Picks + confidence
    kills_pick = pick_from_edge(kill_edge, kills_threshold_used)
    hs_pick = pick_from_edge(hs_edge, hs_threshold_used)

    kills_conf = confidence_from_edge(kill_edge, kills_threshold_used)
    hs_conf = confidence_from_edge(hs_edge, hs_threshold_used)

    with colB:
        st.markdown("### Outputs")
        m1, m2, m3 = st.columns(3)
        m1.metric("Base kills", fmt_num(base_kills, 2))
        m2.metric("Adjusted kills", fmt_num(adj_kills, 2))
        m3.metric("Projected headshots", fmt_num(proj_headshots, 2))

        st.markdown("### Thresholds used (PrizePicks)")
        t1, t2 = st.columns(2)
        t1.metric("Kills threshold", fmt_num(kills_threshold_used, 2))
        t2.metric("Headshots threshold", fmt_num(hs_threshold_used, 2))

        st.markdown("### Edges + Picks")
        e1, e2 = st.columns(2)
        e1.metric("Kills edge (proj - line)", fmt_num(kill_edge, 2))
        e2.metric("Headshots edge (proj - line)", fmt_num(hs_edge, 2))

        p1, p2 = st.columns(2)
        p1.metric("Kills pick", kills_pick)
        p2.metric("Headshots pick", hs_pick)

        c1, c2 = st.columns(2)
        c1.metric("Kills confidence", kills_conf)
        c2.metric("Headshots confidence", hs_conf)

        st.markdown("### Breakdown")
        st.write({
            "Format": fmt,
            "Expected maps": expected_maps,
            "KPM": kpm,
            "HS%": fmt_pct(hs_pct),
            "Core adj total": fmt_pct(role_adj + opp_adj + map_adj),
            "Blowout adj": fmt_pct(blowout_adj),
            "Competitiveness adj": fmt_pct(competitive_adj),
            "Total adj": fmt_pct(total_adj),
            "Role certainty bump": bump,
        })

        # Correlation flag
        st.divider()
        if kills_pick in ("MORE", "LESS") and hs_pick in ("MORE", "LESS"):
            st.warning(
                "Correlation warning: you have BOTH Kills and Headshots picked for the same player. "
                "On PrizePicks this is usually high-correlation. Consider taking only the higher-confidence one "
                "or pairing with a different match."
            )

        st.markdown("### Save to History")
        if st.button("Save this prop to history", type="primary", use_container_width=True):
            ts = datetime.now().isoformat(timespec="seconds")
            row = {
                "timestamp": ts,
                "date": str(d),
                "match": match,
                "player": player,
                "format": fmt,
                "expected_maps": expected_maps,
                "kpm": kpm,
                "hs_pct": hs_pct,

                "role_adj": role_adj,
                "opp_adj": opp_adj,
                "map_adj": map_adj,

                "blowout_risk": blowout_risk,
                "competitiveness": competitiveness,
                "role_certainty": role_certainty,
                "blowout_adj": blowout_adj,
                "competitive_adj": competitive_adj,
                "threshold_bump": bump,

                "total_adj": total_adj,
                "kills_line": kills_line,
                "hs_line": hs_line,

                "kills_threshold_used": kills_threshold_used,
                "hs_threshold_used": hs_threshold_used,

                "base_kills": base_kills,
                "adj_kills": adj_kills,
                "proj_headshots": proj_headshots,
                "kill_edge": kill_edge,
                "hs_edge": hs_edge,
                "kills_pick": kills_pick,
                "hs_pick": hs_pick,
                "kills_confidence": kills_conf,
                "hs_confidence": hs_conf,

                "notes": notes,

                "actual_kills": "",
                "actual_headshots": "",
                "kills_result": "",
                "hs_result": "",
            }
            save_row(row)
            st.success("Saved.")

# ----------------------------
# Tab: History
# ----------------------------
with tabs[1]:
    st.subheader("Saved props + grading (PrizePicks)")

    if history.empty:
        st.info("No history yet. Save a prop from the Model tab.")
    else:
        df = history.copy()

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            player_filter = st.text_input("Filter: player contains", "")
        with f2:
            pick_filter = st.selectbox("Filter: pick", ["All", "MORE", "LESS", "PASS"], index=0)
        with f3:
            conf_filter = st.selectbox("Filter: confidence", ["All", "High", "Medium", "Low"], index=0)
        with f4:
            sort_by = st.selectbox("Sort by", ["Newest", "Kills edge", "HS edge"], index=0)

        if player_filter:
            df = df[df["player"].fillna("").str.contains(player_filter, case=False, na=False)]

        if pick_filter != "All":
            df = df[(df["kills_pick"] == pick_filter) | (df["hs_pick"] == pick_filter)]

        if conf_filter != "All":
            df = df[(df["kills_confidence"] == conf_filter) | (df["hs_confidence"] == conf_filter)]

        if sort_by == "Newest":
            df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("_ts", ascending=False).drop(columns=["_ts"])
        elif sort_by == "Kills edge":
            df["kill_edge"] = pd.to_numeric(df["kill_edge"], errors="coerce")
            df = df.sort_values("kill_edge", ascending=False)
        else:
            df["hs_edge"] = pd.to_numeric(df["hs_edge"], errors="coerce")
            df = df.sort_values("hs_edge", ascending=False)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Grade the most recent saved prop")
        st.caption("This edits only the newest row. For bulk edits, open data/history.csv in Excel/Sheets.")

        latest = load_history().copy()
        latest["_ts"] = pd.to_datetime(latest["timestamp"], errors="coerce")
        latest = latest.sort_values("_ts", ascending=False).drop(columns=["_ts"])
        latest_row = latest.iloc[0].to_dict()

        st.write({
            "timestamp": latest_row.get("timestamp"),
            "player": latest_row.get("player"),
            "match": latest_row.get("match"),
            "kills_pick": latest_row.get("kills_pick"),
            "hs_pick": latest_row.get("hs_pick"),
            "kills_line": latest_row.get("kills_line"),
            "hs_line": latest_row.get("hs_line"),
            "kills_confidence": latest_row.get("kills_confidence"),
            "hs_confidence": latest_row.get("hs_confidence"),
        })

        r1, r2, r3 = st.columns(3)
        with r1:
            actual_kills = st.number_input(
                "Actual kills",
                value=safe_float(latest_row.get("actual_kills"), 0.0),
                step=1.0,
                min_value=0.0,
            )
        with r2:
            actual_hs = st.number_input(
                "Actual headshots",
                value=safe_float(latest_row.get("actual_headshots"), 0.0),
                step=1.0,
                min_value=0.0,
            )
        with r3:
            if st.button("Save results to latest row", use_container_width=True):
                h = load_history()
                h["_ts"] = pd.to_datetime(h["timestamp"], errors="coerce")
                h = h.sort_values("_ts", ascending=False)

                h.loc[h.index[0], "actual_kills"] = actual_kills
                h.loc[h.index[0], "actual_headshots"] = actual_hs

                # compute outcomes
                kpick = h.loc[h.index[0], "kills_pick"]
                hpick = h.loc[h.index[0], "hs_pick"]
                kline = pd.to_numeric(h.loc[h.index[0], "kills_line"], errors="coerce")
                hline = pd.to_numeric(h.loc[h.index[0], "hs_line"], errors="coerce")

                h.loc[h.index[0], "kills_result"] = compute_results(kpick, actual_kills, kline)
                h.loc[h.index[0], "hs_result"] = compute_results(hpick, actual_hs, hline)

                # write back
                h = h.sort_values("_ts", ascending=True).drop(columns=["_ts"])
                h.to_csv(HISTORY_PATH, index=False)

                st.success("Updated latest row with results.")
                st.rerun()

        st.divider()
        st.markdown("### Performance snapshot (rows with results)")

        perf = load_history().copy()
        # Normalize
        for col in ["actual_kills", "actual_headshots", "kills_line", "hs_line"]:
            perf[col] = pd.to_numeric(perf[col], errors="coerce")

        kills_rows = perf[perf["kills_result"].isin(["WIN", "LOSS", "PUSH"])].copy()
        hs_rows = perf[perf["hs_result"].isin(["WIN", "LOSS", "PUSH"])].copy()

        def hit_rate(series: pd.Series) -> float:
            wins = (series == "WIN").sum()
            losses = (series == "LOSS").sum()
            denom = wins + losses
            return (wins / denom) if denom else 0.0

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Kills picks**")
            if kills_rows.empty:
                st.write("No graded kills picks yet.")
            else:
                st.write(kills_rows["kills_result"].value_counts().to_dict())
                st.metric("Hit rate (WIN / (WIN+LOSS))", fmt_pct(hit_rate(kills_rows["kills_result"])))
        with cB:
            st.markdown("**Headshots picks**")
            if hs_rows.empty:
                st.write("No graded headshot picks yet.")
            else:
                st.write(hs_rows["hs_result"].value_counts().to_dict())
                st.metric("Hit rate (WIN / (WIN+LOSS))", fmt_pct(hit_rate(hs_rows["hs_result"])))

# ----------------------------
# Tab: Settings / Help
# ----------------------------
with tabs[2]:
    st.subheader("PrizePicks workflow + what to paste")

    st.markdown(
        """
### What you paste in (manual is best)
**HLTV**
- **KPM** (kills per map) for your chosen time window

**Tracker.gg / CSStats**
- **HS%** as decimal (`47%` → `0.47`)

**BO3.gg + your own judgment**
- role / opponent / map adjustments (small %)

### PrizePicks-specific controls you added
- **Blowout risk**: fewer rounds → lowers kills + headshots
- **Competitiveness**: closer match → increases round volume
- **Role certainty**: if uncertain, thresholds widen by **+1**

### Picks
- **MORE** if edge ≥ threshold
- **LESS** if edge ≤ -threshold
- else **PASS**
- Confidence is based on how far past the threshold you are (High/Medium/Low)

### File
History saves to:
- `data/history.csv`
        """
    )

    st.divider()
    st.markdown("### Install + Run")
    st.code(
        """pip install streamlit pandas
streamlit run app.py
""",
        language="bash",
    )

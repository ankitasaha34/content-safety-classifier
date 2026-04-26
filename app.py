import streamlit as st
import pandas as pd
import numpy as np
import pickle
import anthropic
import os
import json
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# ── Load saved ML model and vectorizer ───────────────────────────
@st.cache_resource
def load_model():
    with open("models/lr_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ── Helper functions ──────────────────────────────────────────────
def ml_bucket(prob):
    if prob >= 0.8: return "HIGH"
    if prob >= 0.5: return "MED"
    if prob >= 0.2: return "LOW"
    return "SAFE"

def action_from_bucket(bucket):
    if bucket == "HIGH": return "🚫 AUTO_BLOCK"
    if bucket == "MED": return "👤 HUMAN_REVIEW"
    if bucket == "LOW": return "⬇️ DOWNRANK"
    return "✅ ALLOW"

def rate_with_claude(comment_text):
    prompt = f"""You are a Trust & Safety analyst reviewing user-generated content for policy violations.

Analyze the following comment and return a JSON response with exactly these fields:

{{
    "policy_category": one of ["hate_speech", "harassment", "sexual_content", "self_harm", "spam", "fraud", "safe"],
    "severity": one of ["none", "low", "medium", "high", "critical"],
    "action": one of ["allow", "downrank", "human_review", "auto_block"],
    "reasoning": "one sentence explanation"
}}

Comment to analyze:
\"\"\"{comment_text}\"\"\"

Return only the JSON object, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def severity_color(severity):
    colors = {
        "none": "🟢", "low": "🟡", 
        "medium": "🟠", "high": "🔴", "critical": "🔴"
    }
    return colors.get(severity, "⚪")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Safety Classifier",
    page_icon="🛡️",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────
st.title("🛡️ AI Content Safety Classifier")
st.markdown("**Trust & Safety pipeline combining ML and Claude API for content moderation**")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Classifier", "📊 Analysis Dashboard", "📋 Case Review Queue"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Live Classifier
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Paste a comment to analyze")
    
    examples = {
        "Harassment example": "You are the stupidest person alive. I hope something terrible happens to you.",
        "Borderline example": "This policy is absolute garbage and whoever made it is an idiot.",
        "Safe example": "I really enjoyed this article, thanks for sharing your perspective."
    }

    selected_example = st.selectbox("Or pick an example:", [""] + list(examples.keys()))
    default_text = examples[selected_example] if selected_example else ""

    comment_input = st.text_area(
        "Comment text",
        value=default_text,
        height=150,
        placeholder="Type or paste a user comment here..."
    )
    if st.button("🔍 Analyze Comment", type="primary"):
        if not comment_input.strip():
            st.warning("Please enter a comment to analyze.")
        else:
            with st.spinner("Running ML model and Claude API..."):
                # ML prediction
                tfidf = vectorizer.transform([comment_input])
                ml_prob = model.predict_proba(tfidf)[0][1]
                bucket = ml_bucket(ml_prob)
                action = action_from_bucket(bucket)

                # Claude prediction
                try:
                    claude_result = rate_with_claude(comment_input)
                    claude_ok = True
                except Exception as e:
                    claude_ok = False
                    claude_error = str(e)

            st.markdown("---")
            st.subheader("Results")

            # ML results
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🤖 ML Model")
                st.metric("Toxicity Probability", f"{ml_prob:.1%}")
                st.metric("Risk Bucket", bucket)
                st.metric("Action", action)

            # Claude results
            with col2:
                st.markdown("### 🧠 Claude API")
                if claude_ok:
                    st.metric("Policy Category", claude_result["policy_category"])
                    st.metric("Severity", f"{severity_color(claude_result['severity'])} {claude_result['severity']}")
                    st.metric("Action", claude_result["action"])
                    st.info(f"**Reasoning:** {claude_result['reasoning']}")
                else:
                    st.error(f"Claude API error: {claude_error}")

            # Disagreement alert
            if claude_ok:
                ml_simple = "flag" if bucket in ["MED", "HIGH"] else "allow"
                claude_simple = "flag" if claude_result["action"] in ["human_review", "auto_block"] else "allow"

                if bucket == "HIGH" and claude_result["action"] in ["human_review", "downrank"]:
                    st.warning("⚠️ **Potential overblocking** — ML says AUTO_BLOCK but Claude recommends human review first.")
                elif claude_result["action"] == "auto_block" and bucket in ["SAFE", "LOW", "MED"]:
                    st.error("🚨 **ML underestimating risk** — Claude recommends AUTO_BLOCK but ML flagged lower severity.")
                elif ml_simple != claude_simple:
                    st.warning("⚠️ **ML and Claude disagree** — sending to human review.")
                else:
                    st.success("✅ ML and Claude agree on this comment.")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis Dashboard
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("System Performance — 200 Comment Sample")
    
    try:
        df = pd.read_csv("models/df_combined.csv")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analyzed", len(df))
        col2.metric("Disagreement Rate", f"{df['disagree'].mean():.1%}")
        col3.metric("Critical Escape Rate", "10.0%")
        col4.metric("Claude Saves", "28")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.image("models/charts/distribution.png", 
                     caption="Policy Category & ML Risk Distribution")
        with col2:
            st.image("models/charts/disagreement.png", 
                     caption="ML vs Claude Agreement Heatmap")

        st.image("models/charts/summary.png", 
                 caption="Full System Summary", width=1200)
    except FileNotFoundError:
        st.error("Run notebooks 01, 02, 03 first to generate the analysis data.")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Case Review Queue
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Human Review Queue — ML vs Claude Disagreements")
    st.markdown("These are cases where ML and Claude disagree — highest priority for human review.")

    try:
        df_disagree = pd.read_csv("models/df_disagree.csv")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            ml_filter = st.multiselect(
                "Filter by ML bucket",
                options=["SAFE", "LOW", "MED", "HIGH"],
                default=["HIGH", "MED"]
            )
        with col2:
            label_filter = st.multiselect(
                "Filter by true label",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda x: "Toxic" if x == 1 else "Safe"
            )

        filtered = df_disagree[
            (df_disagree["ml_bucket"].isin(ml_filter)) &
            (df_disagree["true_label"].isin(label_filter))
        ]

        st.markdown(f"**Showing {len(filtered)} cases**")

        for _, row in filtered.head(20).iterrows():
            with st.expander(
                f"[ML: {row['ml_bucket']}] [Claude: {row.get('claude_bucket','?')}] "
                f"{'🔴 TOXIC' if row['true_label']==1 else '🟢 SAFE'} — "
                f"{str(row['comment_text'])[:80]}..."
            ):
                st.write("**Full comment:**", row["comment_text"])
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**ML probability:**", round(row["ml_prob"], 3))
                    st.write("**ML bucket:**", row["ml_bucket"])
                with col2:
                    st.write("**Claude category:**", row.get("claude_category", "N/A"))
                    st.write("**Claude severity:**", row.get("claude_severity", "N/A"))
                    st.write("**Claude reasoning:**", row.get("claude_reasoning", "N/A"))

    except FileNotFoundError:
        st.error("Run notebooks 01, 02, 03 first to generate the disagreement data.")
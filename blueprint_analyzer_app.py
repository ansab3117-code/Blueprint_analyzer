"""
AI-Powered Blueprint Analysis System — Streamlit Demo
Uses OpenRouter API with Google Gemini Flash (document-capable model)
"""

import streamlit as st
import requests
import json
import base64
import time
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blueprint AI Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e4dc;
  }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1f2330;
  }
  section[data-testid="stSidebar"] label { color: #a0a8b8 !important; font-size: 0.82rem; }

  /* Main container */
  .block-container { padding: 2rem 2.5rem; }

  /* Cards */
  .card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .card-accent { border-left: 3px solid #f5a623; }

  /* Metric boxes */
  .metric-box {
    background: #0d0f14;
    border: 1px solid #1f2330;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f5a623;
  }
  .metric-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }

  /* Tables */
  .styled-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  .styled-table th {
    background: #1a1e29;
    color: #f5a623;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.6rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid #1f2330;
  }
  .styled-table td { padding: 0.55rem 0.8rem; border-bottom: 1px solid #141720; color: #cdd0d8; }
  .styled-table tr:hover td { background: #161a24; }

  /* Highlight tag */
  .tag {
    display: inline-block;
    background: #1e2133;
    color: #f5a623;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    margin-right: 4px;
  }

  /* Buttons */
  .stButton > button {
    background: #f5a623 !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.04em;
  }
  .stButton > button:hover { background: #e09415 !important; }

  /* Inputs */
  .stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #13161e !important;
    border: 1px solid #1f2330 !important;
    color: #e8e4dc !important;
    border-radius: 6px !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #13161e !important;
    border: 1px solid #1f2330 !important;
    color: #e8e4dc !important;
    border-radius: 6px !important;
  }

  /* Status indicators */
  .status-ok { color: #4ade80; font-weight: 600; }
  .status-warn { color: #f5a623; font-weight: 600; }
  .status-err { color: #f87171; font-weight: 600; }

  /* Section header */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #f5a623;
    border-bottom: 1px solid #1f2330;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Progress bar */
  .stProgress > div > div { background-color: #f5a623 !important; }

  /* Divider */
  hr { border-color: #1f2330 !important; }

  /* Alert override */
  .stAlert { border-radius: 6px !important; }

  /* JSON viewer */
  .stJson { background: #13161e !important; border: 1px solid #1f2330 !important; border-radius: 6px !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Best document-capable models on OpenRouter
MODELS = {
    "Google Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
    "Google Gemini 1.5 Pro": "google/gemini-1.5-pro",
    "Anthropic Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "OpenAI GPT-4o": "openai/gpt-4o",
    "Meta Llama 3.2 90B Vision": "meta-llama/llama-3.2-90b-vision-instruct",
}

ANALYSIS_MODULES = {
    "Floor Plan & Room Schedule": "floor_plan",
    "Surface Area Calculation": "surface_area",
    "Paint & Material Quantities": "paint_material",
    "Labor Estimation": "labor",
    "Full Cost Summary": "cost_summary",
}

# ── Helper functions ───────────────────────────────────────────────────────────

def encode_pdf_to_base64(uploaded_file) -> str:
    """Encode uploaded PDF to base64 string."""
    return base64.standard_b64encode(uploaded_file.read()).decode("utf-8")


def build_analysis_prompt(module: str, extra_context: str = "") -> str:
    """Return a structured extraction prompt based on analysis module."""
    base = (
        "You are an expert architectural estimator and construction cost analyst. "
        "You are analyzing a multi-page blueprint PDF for residential construction. "
        "Extract data carefully, be specific with numbers, and format your response "
        "in clearly structured sections with tables where applicable.\n\n"
    )

    prompts = {
        "floor_plan": (
            base +
            "TASK — FLOOR PLAN & ROOM SCHEDULE EXTRACTION:\n"
            "1. Identify the plan name, elevation code, and revision date.\n"
            "2. List every room/space found on each floor with its approximate area (sq.ft.) "
            "and ceiling height if annotated.\n"
            "3. Note any optional or alternate layouts detected.\n"
            "4. List structural annotations: wall types, beam sizes, joist depths.\n"
            "5. Summarize total floor area per level.\n"
            "Present results in structured Markdown tables."
        ),
        "surface_area": (
            base +
            "TASK — SURFACE AREA CALCULATION:\n"
            "For each room identified, estimate:\n"
            "1. Interior wall area (sq.ft.) = perimeter × ceiling height − door/window openings\n"
            "   - Standard interior door deduction: 21 sq.ft.; exterior door: 24.5 sq.ft.\n"
            "   - Apply 1.15 waste/overlap multiplier for 2-coat paint systems\n"
            "2. Ceiling area (sq.ft.) — flat × 1.0, vaulted × 1.25, tray adds side area\n"
            "3. Trim/casings allowance: 10% of wall area in lin.ft.\n"
            "4. Basement development wall areas (8 ft ceiling default)\n"
            "5. Exterior wall area estimate\n"
            "Provide a summary table: Zone | Floor Area | Wall Area | Ceiling Area | Notes."
        ),
        "paint_material": (
            base +
            "TASK — PAINT & FINISH MATERIAL QUANTITIES:\n"
            "Using surface areas from the blueprint, calculate gallons required for:\n"
            "- Interior wall primer (PVA flat, 350 sq.ft./gal, 1 coat, 10% waste)\n"
            "- Interior wall finish (eggshell, 400 sq.ft./gal, 2 coats, 8% waste)\n"
            "- Ceiling paint (flat white, 400 sq.ft./gal, 2 coats, 5% waste)\n"
            "- Trim & casings (semi-gloss, 450 sq.ft./gal, 2 coats, 12% waste)\n"
            "- Exterior primer (acrylic block, 300 sq.ft./gal, 1 coat, 12% waste)\n"
            "- Exterior finish (satin, 350 sq.ft./gal, 2 coats, 10% waste)\n"
            "- Basement dev. walls (eggshell, 380 sq.ft./gal, 2 coats, 8% waste)\n"
            "Formula: Gallons = (Area × Coats × Waste Factor) ÷ Coverage Rate\n"
            "Show a table: Item | Area (sq.ft.) | Gallons | Unit Cost | Subtotal.\n"
            "Use current typical USD pricing for residential paint products."
        ),
        "labor": (
            base +
            "TASK — LABOR ESTIMATION BY TRADE:\n"
            "Estimate labor hours and costs for the following trades using these productivity rates:\n"
            "- Painter interior: 200 sq.ft./hr @ ~$48/hr\n"
            "- Painter exterior: 150 sq.ft./hr @ ~$52/hr\n"
            "- Finish carpenter (trim): 30 lin.ft./hr @ ~$62/hr\n"
            "- Framer: 8 sq.ft./hr @ ~$58/hr\n"
            "- Drywaller (hang+tape): 80 sq.ft./hr @ ~$55/hr\n"
            "- Electrician (rough-in): 0.75 hr/device @ ~$88/hr\n"
            "- Plumber (rough-in): 4 hrs/fixture @ ~$95/hr\n"
            "- HVAC technician: 0.05 hr/sq.ft. @ ~$78/hr\n"
            "- Insulator: 0.04 hr/sq.ft. exterior wall @ ~$52/hr\n"
            "- Flooring installer: 0.06 hr/sq.ft. @ ~$48/hr\n"
            "Estimate quantities from the blueprint and compute hours × rate.\n"
            "Present: Trade | Quantity | Hours | Rate | Labor Cost.\n"
            "Provide a TOTAL direct labor cost."
        ),
        "cost_summary": (
            base +
            "TASK — FULL PROJECT COST SUMMARY:\n"
            "Produce a comprehensive cost build-up covering all major categories:\n"
            "1. Paint & finish materials\n"
            "2. Framing lumber & sheathing\n"
            "3. Drywall materials\n"
            "4. Doors & windows (from schedule)\n"
            "5. Rough mechanical (electrical, plumbing, HVAC)\n"
            "6. Insulation\n"
            "7. Flooring\n"
            "8. Cabinetry & millwork\n"
            "9. Direct labor (all trades)\n"
            "10. General conditions & site (5% of subtotal)\n"
            "11. Builder overhead & profit (12% of subtotal)\n"
            "12. Contingency reserve (8% of subtotal)\n\n"
            "Then provide a VARIANT COMPARISON TABLE showing estimated totals for different "
            "build configurations (base only, with basement dev., with upper floor option, etc.).\n\n"
            "Format: Category | Basis | Amount (USD) | % of Total.\n"
            "End with a GRAND TOTAL and key insights for the builder."
        ),
    }

    prompt = prompts.get(module, base + "Analyze this blueprint and provide a detailed construction estimate.")
    if extra_context.strip():
        prompt += f"\n\nADDITIONAL CONTEXT FROM USER:\n{extra_context}"
    return prompt


def call_openrouter(api_key: str, model: str, pdf_b64: str, prompt: str) -> dict:
    """Call OpenRouter API with PDF document."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://blueprint-ai-analyzer.demo",
        "X-Title": "Blueprint AI Analyzer",
    }

    # Build message with PDF document
    content = [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:application/pdf;base64,{pdf_b64}",
            },
        },
    ]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    return response


def parse_response(response) -> tuple[bool, str]:
    """Parse API response and return (success, content/error)."""
    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return True, content
    else:
        try:
            err = response.json()
            return False, f"API Error {response.status_code}: {err.get('error', {}).get('message', str(err))}"
        except Exception:
            return False, f"HTTP Error {response.status_code}: {response.text[:400]}"


def display_metrics_row(metrics: list[tuple[str, str]]):
    """Render a row of metric boxes."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏗️ Blueprint AI")
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    selected_model_name = st.selectbox(
        "AI Model",
        list(MODELS.keys()),
        index=0,
        help="Gemini 2.0 Flash is recommended — fast, cheap, and handles PDFs natively.",
    )
    selected_model = MODELS[selected_model_name]

    st.markdown('<div class="section-header">Analysis Module</div>', unsafe_allow_html=True)
    selected_module_name = st.selectbox(
        "What to Analyze",
        list(ANALYSIS_MODULES.keys()),
        index=4,
    )
    selected_module = ANALYSIS_MODULES[selected_module_name]

    extra_context = st.text_area(
        "Additional Instructions (optional)",
        placeholder="e.g. Focus on the basement suite option. Use Canadian Prairie labor rates.",
        height=90,
    )

    st.markdown("---")
    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#6b7280; line-height:1.6;">
    Demo for the <b style="color:#a0a8b8">AI-Powered Blueprint Analysis System</b>.<br><br>
    Supports multi-page PDFs. Extracts floor plans, surface areas, paint quantities, labor hours, and cost estimates using AI vision models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.75rem; color:#374151; margin-top:1.2rem;">
    Model: <span style="color:#f5a623; font-family:monospace;">{}</span>
    </div>
    """.format(selected_model), unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:0.3rem;">
  <span style="font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.2em;
               text-transform:uppercase; color:#f5a623;">System v1.0 — April 2026</span>
</div>
<h1 style="margin:0; font-size:2rem; line-height:1.15;">
  AI Blueprint<br>Analysis System
</h1>
<p style="color:#6b7280; margin-top:0.5rem; font-size:0.92rem; max-width:560px;">
  Upload a residential construction blueprint PDF. The AI extracts floor-plan geometry,
  room schedules, surface areas, material quantities, and project cost estimates.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── API Key Input ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">OpenRouter API Key</div>', unsafe_allow_html=True)

col_key, col_key_hint = st.columns([2, 1], gap="large")
with col_key:
    api_key = st.text_input(
        "api_key_main",
        type="password",
        placeholder="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        help="Your key is never stored — it's used only for the current request.",
    )
with col_key_hint:
    st.markdown("""
    <div style="padding-top:0.45rem; font-size:0.8rem; color:#6b7280; line-height:1.6;">
      Get a free key at
      <a href="https://openrouter.ai/keys" target="_blank"
         style="color:#f5a623; text-decoration:none;">openrouter.ai/keys</a>.<br>
      <span style="font-size:0.75rem; color:#374151;">Not stored · used only for this request.</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Upload Section ─────────────────────────────────────────────────────────────

col_upload, col_info = st.columns([2, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-header">Upload Blueprint PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop blueprint PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        help="Multi-page blueprint PDFs supported. Max ~20MB recommended.",
    )

    if uploaded_file:
        st.markdown(f"""
        <div class="card card-accent">
          <span class="tag">PDF</span>
          <span style="font-family:'Space Mono',monospace; font-size:0.85rem;">{uploaded_file.name}</span>
          <span style="color:#6b7280; font-size:0.78rem; margin-left:0.8rem;">
            {uploaded_file.size / 1024:.1f} KB
          </span>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="section-header">Reference Plans</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <div style="font-size:0.78rem; color:#6b7280; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.07em;">Tested With</div>
      <div style="font-family:'Space Mono',monospace; font-size:0.8rem; color:#e8e4dc; line-height:2;">
        📐 Grada 2025 Modern Farmhouse<br>
        📐 Almer 2025 Prairie<br>
      </div>
      <div style="font-size:0.75rem; color:#4b5563; margin-top:0.6rem;">
        Hotchkiss stock plan family
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Run Analysis ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Analysis Controls</div>', unsafe_allow_html=True)

col_btn, col_hint = st.columns([1, 3])
with col_btn:
    run_btn = st.button("⚡ Run Analysis", use_container_width=True)
with col_hint:
    st.markdown(f"""
    <div style="padding-top:0.6rem; font-size:0.82rem; color:#6b7280;">
      Module: <span class="tag">{selected_module_name}</span>
      Model: <span class="tag">{selected_model_name.split()[0]} {selected_model_name.split()[1] if len(selected_model_name.split()) > 1 else ''}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Analysis Results ───────────────────────────────────────────────────────────

if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
    elif not uploaded_file:
        st.error("⚠️ Please upload a blueprint PDF.")
    else:
        st.markdown("---")
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.markdown('<span class="status-warn">⟳ Encoding PDF document...</span>', unsafe_allow_html=True)
            progress_bar.progress(15)

            # Read and encode PDF — seek to start first in case already read
            uploaded_file.seek(0)
            pdf_b64 = encode_pdf_to_base64(uploaded_file)

            status_text.markdown('<span class="status-warn">⟳ Building analysis prompt...</span>', unsafe_allow_html=True)
            progress_bar.progress(30)

            prompt = build_analysis_prompt(selected_module, extra_context)

            status_text.markdown(f'<span class="status-warn">⟳ Calling {selected_model_name} via OpenRouter...</span>', unsafe_allow_html=True)
            progress_bar.progress(50)

            t0 = time.time()
            response = call_openrouter(api_key, selected_model, pdf_b64, prompt)
            elapsed = time.time() - t0

            progress_bar.progress(90)
            success, content = parse_response(response)

            if success:
                progress_bar.progress(100)
                status_text.markdown(f'<span class="status-ok">✓ Analysis complete — {elapsed:.1f}s</span>', unsafe_allow_html=True)

                # Metrics row
                token_data = response.json().get("usage", {})
                prompt_tokens = token_data.get("prompt_tokens", "—")
                completion_tokens = token_data.get("completion_tokens", "—")

                st.markdown("<br>", unsafe_allow_html=True)
                display_metrics_row([
                    ("Analysis Module", selected_module_name.split()[0]),
                    ("Response Time", f"{elapsed:.1f}s"),
                    ("Prompt Tokens", str(prompt_tokens)),
                    ("Output Tokens", str(completion_tokens)),
                ])
                st.markdown("<br>", unsafe_allow_html=True)

                # Main result card
                st.markdown(f"""
                <div class="card" style="border-top: 2px solid #f5a623;">
                  <div style="font-family:'Space Mono',monospace; font-size:0.68rem; 
                               letter-spacing:0.12em; text-transform:uppercase;
                               color:#f5a623; margin-bottom:1rem;">
                    {selected_module_name} — {selected_model_name}
                  </div>
                """, unsafe_allow_html=True)

                st.markdown(content)
                st.markdown("</div>", unsafe_allow_html=True)

                # Expandable: raw JSON
                with st.expander("🔍 Raw API Response (JSON)"):
                    st.json(response.json())

                # Download button
                st.download_button(
                    label="⬇ Download Analysis (Markdown)",
                    data=f"# {selected_module_name}\n\nModel: {selected_model_name}\nTime: {elapsed:.1f}s\n\n---\n\n{content}",
                    file_name=f"blueprint_analysis_{selected_module}.md",
                    mime="text/markdown",
                )

            else:
                progress_bar.progress(100)
                status_text.markdown('<span class="status-err">✗ API call failed</span>', unsafe_allow_html=True)
                st.error(content)

                st.markdown("""
                **Troubleshooting tips:**
                - Verify your OpenRouter API key is correct and has credits
                - Some models (e.g. Llama Vision) may not support PDF input — try Gemini 2.0 Flash
                - Check [openrouter.ai/activity](https://openrouter.ai/activity) for usage logs
                """)

        except requests.exceptions.Timeout:
            progress_bar.progress(100)
            status_text.markdown('<span class="status-err">✗ Request timed out</span>', unsafe_allow_html=True)
            st.error("The request timed out after 120 seconds. Try a smaller PDF or a faster model.")

        except Exception as e:
            progress_bar.progress(100)
            status_text.markdown('<span class="status-err">✗ Unexpected error</span>', unsafe_allow_html=True)
            st.exception(e)

# ── Info cards when no analysis running ───────────────────────────────────────
else:
    st.markdown("---")
    st.markdown('<div class="section-header">System Capabilities</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
          <div style="font-size:1.4rem; margin-bottom:0.5rem;">📐</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.82rem; color:#f5a623; margin-bottom:0.4rem;">FLOOR PLAN PARSING</div>
          <div style="font-size:0.82rem; color:#9ca3af; line-height:1.6;">
            Extracts room schedules, dimensions, ceiling heights, and structural annotations from multi-page blueprint PDFs.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <div style="font-size:1.4rem; margin-bottom:0.5rem;">🎨</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.82rem; color:#f5a623; margin-bottom:0.4rem;">PAINT QUANTITIES</div>
          <div style="font-size:0.82rem; color:#9ca3af; line-height:1.6;">
            Computes gallons by surface type using industry coverage rates, coat counts, and waste factors for interior and exterior finishes.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
          <div style="font-size:1.4rem; margin-bottom:0.5rem;">💰</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.82rem; color:#f5a623; margin-bottom:0.4rem;">COST ESTIMATES</div>
          <div style="font-size:0.82rem; color:#9ca3af; line-height:1.6;">
            Full project cost build-up: materials, labor by trade, overhead, and contingency. Variant comparison across build configurations.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="section-header" style="margin-top:1.8rem;">Processing Pipeline</div>', unsafe_allow_html=True)

    stages = [
        ("01", "INGEST", "PDF parsed, pages classified"),
        ("02", "EXTRACT", "AI reads floor plans & dimensions"),
        ("03", "GEOMETRY", "Surface areas computed per room"),
        ("04", "COST CALC", "Rate cards applied to all quantities"),
        ("05", "REPORT", "Structured output with tables"),
    ]

    cols = st.columns(len(stages))
    for col, (num, title, desc) in zip(cols, stages):
        col.markdown(f"""
        <div style="text-align:center; padding:1rem 0.5rem;">
          <div style="font-family:'Space Mono',monospace; font-size:1.5rem; color:#1f2330; 
                       font-weight:700; line-height:1;">{num}</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#f5a623; 
                       margin:0.4rem 0 0.3rem; letter-spacing:0.1em;">{title}</div>
          <div style="font-size:0.75rem; color:#4b5563; line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sample cost table
    st.markdown('<div class="section-header" style="margin-top:1.8rem;">Sample Output — Grada 2025 Cost Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="styled-table">
      <thead>
        <tr><th>#</th><th>Cost Category</th><th>Amount (USD)</th><th>% of Total</th></tr>
      </thead>
      <tbody>
        <tr><td>1</td><td>Paint & Finish Materials</td><td>$4,300</td><td>3.2%</td></tr>
        <tr><td>2</td><td>Framing Lumber & Sheathing</td><td>$18,500</td><td>13.8%</td></tr>
        <tr><td>3</td><td>Drywall Materials</td><td>$6,200</td><td>4.6%</td></tr>
        <tr><td>4</td><td>Doors & Windows</td><td>$14,800</td><td>11.0%</td></tr>
        <tr><td>5</td><td>Rough Mechanical</td><td>$22,400</td><td>16.7%</td></tr>
        <tr><td>6–8</td><td>Insulation + Flooring + Cabinetry</td><td>$33,300</td><td>24.8%</td></tr>
        <tr><td>9</td><td>Direct Labor (all trades, 967 hrs)</td><td>$59,184</td><td>44.0%</td></tr>
        <tr><td>10–12</td><td>G.C. + Overhead + Contingency</td><td>$39,696</td><td>29.6%</td></tr>
        <tr style="background:#1a1e29;">
          <td colspan="2"><b style="color:#f5a623;">TOTAL ESTIMATED PROJECT COST</b></td>
          <td><b style="color:#f5a623;">$198,380</b></td>
          <td><b style="color:#f5a623;">100%</b></td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.caption("Sample data from system spec document — Grada 2025 Base + 2-Bed Basement Development")

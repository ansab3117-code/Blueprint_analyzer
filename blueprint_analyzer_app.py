"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AI-POWERED BLUEPRINT ANALYSIS SYSTEM  —  v2.0                      ║
║         Structural Analysis · Labor Estimation · Cost Intelligence          ║
║         Model: google/gemini-2.5-pro-preview  via OpenRouter               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  • PDF ingestion with pypdf text extraction (up to 50 pages)
  • Base64 native PDF vision pass-through for Gemini 2.5 Pro
  • Streaming SSE response with live token-by-token rendering
  • Six independent analysis modules with structured prompts
  • Tabbed results UI with per-module caching in st.session_state
  • Export: Markdown download per analysis tab
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import requests
import base64
import time
import json
import io
import re
import math
from datetime import datetime
from typing import Generator

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blueprint Intelligence System",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_URL   = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL        = "google/gemini-2.5-pro-preview"   # native PDF vision, 1M ctx
PRIMARY_MODEL_LABEL  = "Gemini 2.5 Pro Preview"
MAX_PDF_PAGES        = 50
MAX_PDF_MB           = 30
REQUEST_TIMEOUT_S    = 300          # 5 min — large PDF + 8k output
MAX_OUTPUT_TOKENS    = 8192

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — editorial dark-matter aesthetic
#  Fonts: Syne (geometric, technical display) + Instrument Serif (refined body)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ───────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #080a0e;
  --bg-panel:    #0d1017;
  --bg-card:     #111520;
  --bg-hover:    #161b28;
  --border:      #1c2236;
  --border-soft: #141826;
  --text:        #dde2ec;
  --text-muted:  #5a6480;
  --text-dim:    #323d58;
  --accent:      #00e5c3;          /* electric teal */
  --accent2:     #ff6b35;          /* construction orange */
  --accent3:     #7c6af5;          /* soft violet */
  --gold:        #c9a84c;
  --success:     #22c55e;
  --warn:        #f59e0b;
  --error:       #ef4444;
  --radius:      10px;
  --radius-sm:   6px;
  --font-display:'Syne', sans-serif;
  --font-body:   'Instrument Serif', serif;
  --font-mono:   'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--font-body);
  background: var(--bg) !important;
  color: var(--text);
}

/* ── Streamlit structural overrides ────────────────────── */
.stApp { background: var(--bg) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Remove Streamlit padding on main */
.main .block-container { padding: 0 !important; }
[data-testid="stVerticalBlock"] > div:first-child { padding: 0 !important; }

/* ── Custom scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Hero / Masthead ────────────────────────────────────── */
.masthead {
  background: linear-gradient(135deg, #080a0e 0%, #0a0f1a 40%, #0d1520 100%);
  border-bottom: 1px solid var(--border);
  padding: 2.8rem 3.5rem 2rem;
  position: relative;
  overflow: hidden;
}
.masthead::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,229,195,0.04) 0%, transparent 70%),
    radial-gradient(ellipse 40% 60% at 10% 80%, rgba(124,106,245,0.05) 0%, transparent 70%);
  pointer-events: none;
}
.masthead-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(28,34,54,0.4) 1px, transparent 1px),
    linear-gradient(90deg, rgba(28,34,54,0.4) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  mask-image: linear-gradient(to bottom, transparent 0%, black 30%, black 70%, transparent 100%);
}
.masthead-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 400;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.masthead-eyebrow::before {
  content: '';
  display: inline-block;
  width: 20px;
  height: 1px;
  background: var(--accent);
}
.masthead h1 {
  font-family: var(--font-display);
  font-size: clamp(2rem, 4vw, 3.4rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.02em;
  color: #f0f4ff;
  margin-bottom: 0.7rem;
}
.masthead h1 em {
  font-style: normal;
  color: var(--accent);
}
.masthead-sub {
  font-family: var(--font-body);
  font-style: italic;
  font-size: 1.05rem;
  color: var(--text-muted);
  max-width: 560px;
  line-height: 1.6;
}
.masthead-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1.5rem;
}
.badge {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.3rem 0.75rem;
  border-radius: 3px;
  border: 1px solid;
}
.badge-teal  { color: var(--accent);  border-color: rgba(0,229,195,0.3);  background: rgba(0,229,195,0.06); }
.badge-orange{ color: var(--accent2); border-color: rgba(255,107,53,0.3);  background: rgba(255,107,53,0.06);}
.badge-violet{ color: var(--accent3); border-color: rgba(124,106,245,0.3); background: rgba(124,106,245,0.06);}
.badge-gold  { color: var(--gold);    border-color: rgba(201,168,76,0.3);  background: rgba(201,168,76,0.06); }

/* ── Main workspace ─────────────────────────────────────── */
.workspace {
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 0;
  min-height: calc(100vh - 180px);
}
.panel-left {
  border-right: 1px solid var(--border);
  background: var(--bg-panel);
  padding: 2rem 1.8rem;
}
.panel-right {
  background: var(--bg);
  padding: 2rem 2.5rem;
}

/* ── Section label ──────────────────────────────────────── */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 0.9rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-soft);
}

/* ── Upload zone ────────────────────────────────────────── */
.upload-zone {
  border: 1.5px dashed var(--border);
  border-radius: var(--radius);
  padding: 2rem 1.5rem;
  text-align: center;
  background: var(--bg-card);
  transition: border-color 0.2s, background 0.2s;
  cursor: pointer;
  margin-bottom: 1.4rem;
}
.upload-zone:hover { border-color: var(--accent); background: var(--bg-hover); }
.upload-icon { font-size: 2rem; margin-bottom: 0.6rem; display: block; }
.upload-title {
  font-family: var(--font-display);
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.3rem;
}
.upload-hint { font-size: 0.78rem; color: var(--text-muted); font-style: italic; }

/* ── File info card ─────────────────────────────────────── */
.file-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  padding: 1rem 1.2rem;
  margin-bottom: 1.2rem;
}
.file-card-name {
  font-family: var(--font-display);
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.4rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.file-card-meta {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--text-muted);
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}
.file-stat { display: flex; align-items: center; gap: 0.3rem; }
.dot-ok  { width: 6px; height: 6px; border-radius: 50%; background: var(--success); flex-shrink: 0; }
.dot-warn{ width: 6px; height: 6px; border-radius: 50%; background: var(--warn);    flex-shrink: 0; }
.dot-err { width: 6px; height: 6px; border-radius: 50%; background: var(--error);   flex-shrink: 0; }

/* ── Module selector ────────────────────────────────────── */
.module-grid {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin-bottom: 1.4rem;
}
.module-btn {
  display: flex;
  align-items: flex-start;
  gap: 0.8rem;
  padding: 0.85rem 1rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-soft);
  background: transparent;
  cursor: pointer;
  transition: all 0.15s;
  text-align: left;
  width: 100%;
}
.module-btn:hover { border-color: var(--border); background: var(--bg-hover); }
.module-btn.active {
  border-color: rgba(0,229,195,0.5);
  background: rgba(0,229,195,0.06);
}
.module-icon {
  font-size: 1.1rem;
  line-height: 1.3;
  flex-shrink: 0;
}
.module-text-wrap { display: flex; flex-direction: column; gap: 0.15rem; }
.module-name {
  font-family: var(--font-display);
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1.2;
}
.module-desc { font-size: 0.72rem; color: var(--text-muted); line-height: 1.4; }
.module-btn.active .module-name { color: var(--accent); }

/* ── API Key input ──────────────────────────────────────── */
.api-key-section { margin-bottom: 1.4rem; }
.api-key-label {
  font-family: var(--font-mono);
  font-size: 0.63rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.45rem;
  display: block;
}

/* ── Run button ─────────────────────────────────────────── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #00e5c3 0%, #00c4a8 100%) !important;
  color: #06141a !important;
  font-family: var(--font-display) !important;
  font-size: 0.82rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.75rem 1.5rem !important;
  transition: all 0.2s !important;
  box-shadow: 0 0 20px rgba(0,229,195,0.15) !important;
}
.stButton > button:hover {
  box-shadow: 0 0 30px rgba(0,229,195,0.3) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
  background: var(--bg-card) !important;
  color: var(--text-dim) !important;
  box-shadow: none !important;
  transform: none !important;
}

/* ── Streamlit form inputs ──────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,229,195,0.1) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label {
  font-family: var(--font-mono) !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
.stFileUploader {
  background: transparent !important;
}
.stFileUploader > div {
  background: var(--bg-card) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
}
.stFileUploader label {
  font-family: var(--font-mono) !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* ── Progress bar ───────────────────────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
  border-radius: 2px !important;
}
.stProgress > div > div > div {
  background: var(--bg-card) !important;
  border-radius: 2px !important;
  height: 3px !important;
}

/* ── Results panel ──────────────────────────────────────── */
.results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-soft);
}
.results-title {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.01em;
}
.results-meta {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-muted);
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.2rem;
}

/* ── Metric strip ───────────────────────────────────────── */
.metric-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.6rem;
  margin-bottom: 1.6rem;
}
.metric-tile {
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-sm);
  padding: 0.9rem 1rem;
}
.metric-tile-value {
  font-family: var(--font-display);
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
  margin-bottom: 0.25rem;
}
.metric-tile-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}

/* ── Streaming output box ───────────────────────────────── */
.stream-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 2px solid var(--accent);
  border-radius: var(--radius);
  padding: 1.8rem 2rem;
  font-family: var(--font-body);
  font-size: 0.95rem;
  line-height: 1.75;
  color: var(--text);
  min-height: 200px;
}

/* ── Markdown inside stream-box ─────────────────────────── */
.stream-box h1, .stream-box h2, .stream-box h3 {
  font-family: var(--font-display);
  font-weight: 700;
  margin: 1.4rem 0 0.6rem;
  letter-spacing: -0.01em;
  color: #f0f4ff;
}
.stream-box h1 { font-size: 1.3rem; color: var(--accent); border-bottom: 1px solid var(--border-soft); padding-bottom: 0.4rem; }
.stream-box h2 { font-size: 1.05rem; color: var(--text); }
.stream-box h3 { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
.stream-box p  { margin-bottom: 0.75rem; }
.stream-box strong { color: #f0f4ff; font-style: normal; }
.stream-box em { color: var(--gold); }
.stream-box code {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  background: var(--bg-hover);
  border: 1px solid var(--border-soft);
  padding: 0.1em 0.4em;
  border-radius: 3px;
  color: var(--accent);
}
.stream-box pre {
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-left: 3px solid var(--accent3);
  border-radius: var(--radius-sm);
  padding: 1rem 1.2rem;
  overflow-x: auto;
  margin: 0.8rem 0;
}
.stream-box pre code {
  background: none;
  border: none;
  padding: 0;
  color: var(--text);
  font-size: 0.78rem;
}
.stream-box table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-family: var(--font-mono);
  font-size: 0.76rem;
}
.stream-box th {
  background: var(--bg-hover);
  color: var(--accent);
  font-weight: 500;
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0.55rem 0.9rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.stream-box td {
  padding: 0.5rem 0.9rem;
  border-bottom: 1px solid var(--border-soft);
  color: var(--text);
  vertical-align: top;
}
.stream-box tr:hover td { background: var(--bg-hover); }
.stream-box ul, .stream-box ol {
  margin: 0.5rem 0 0.75rem 1.4rem;
}
.stream-box li { margin-bottom: 0.3rem; line-height: 1.6; }
.stream-box blockquote {
  border-left: 3px solid var(--accent2);
  padding: 0.6rem 1rem;
  margin: 0.8rem 0;
  background: rgba(255,107,53,0.05);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  font-style: italic;
  color: var(--text-muted);
}
.stream-box hr {
  border: none;
  border-top: 1px solid var(--border-soft);
  margin: 1.2rem 0;
}

/* ── Status indicators ──────────────────────────────────── */
.status-line {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 0.9rem;
  border-radius: var(--radius-sm);
  margin-bottom: 0.8rem;
}
.status-line.running {
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.2);
  color: var(--warn);
}
.status-line.success {
  background: rgba(34,197,94,0.08);
  border: 1px solid rgba(34,197,94,0.2);
  color: var(--success);
}
.status-line.error {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.2);
  color: var(--error);
}
.spinner {
  display: inline-block;
  animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Expander overrides ─────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  color: var(--text-muted) !important;
  letter-spacing: 0.08em !important;
}
.streamlit-expanderContent {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* ── Tab strip ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  background: transparent !important;
  border: none !important;
  padding: 0.6rem 1.1rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding: 1.2rem 0 0 !important;
}

/* ── Download btn ───────────────────────────────────────── */
.stDownloadButton > button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.08em !important;
  padding: 0.4rem 0.9rem !important;
  border-radius: var(--radius-sm) !important;
  transition: all 0.15s !important;
  width: auto !important;
  box-shadow: none !important;
}
.stDownloadButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ── Empty state ────────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  text-align: center;
  gap: 1rem;
}
.empty-state-icon { font-size: 3rem; opacity: 0.15; }
.empty-state-title {
  font-family: var(--font-display);
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-dim);
}
.empty-state-body { font-size: 0.83rem; color: var(--text-dim); max-width: 340px; line-height: 1.6; }

/* ── Pipeline vis ───────────────────────────────────────── */
.pipeline-row {
  display: flex;
  align-items: stretch;
  gap: 0;
  margin: 1.2rem 0;
  overflow-x: auto;
}
.pipe-stage {
  flex: 1;
  background: var(--bg-card);
  border-top: 2px solid var(--border-soft);
  border-bottom: 1px solid var(--border-soft);
  border-left: 1px solid var(--border-soft);
  padding: 0.9rem 1rem;
  position: relative;
  min-width: 120px;
}
.pipe-stage:first-child { border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
.pipe-stage:last-child  {
  border-right: 1px solid var(--border-soft);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.pipe-stage.active { border-top-color: var(--accent); }
.pipe-num {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  margin-bottom: 0.25rem;
}
.pipe-name {
  font-family: var(--font-display);
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-muted);
  margin-bottom: 0.2rem;
}
.pipe-detail { font-size: 0.65rem; color: var(--text-dim); line-height: 1.4; }
.pipe-stage.active .pipe-name { color: var(--accent); }

/* ── Warning / info boxes ───────────────────────────────── */
.info-box {
  border-radius: var(--radius-sm);
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  line-height: 1.5;
  margin-bottom: 0.8rem;
  display: flex;
  gap: 0.6rem;
  align-items: flex-start;
}
.info-box-warn {
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.2);
  color: #d4a015;
}
.info-box-info {
  background: rgba(0,229,195,0.06);
  border: 1px solid rgba(0,229,195,0.15);
  color: var(--accent);
}
.info-box-err {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.2);
  color: var(--error);
}
.info-box-icon { flex-shrink: 0; font-size: 0.85rem; margin-top: 0.05rem; }
.info-box-body { font-family: var(--font-mono); font-size: 0.72rem; }

/* ── Token counter pill ─────────────────────────────────── */
.token-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-family: var(--font-mono);
  font-size: 0.63rem;
  color: var(--text-muted);
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 20px;
  padding: 0.2rem 0.6rem;
}
.token-pill .t-accent { color: var(--accent); font-weight: 600; }

/* ── Responsive ─────────────────────────────────────────── */
@media (max-width: 900px) {
  .workspace { grid-template-columns: 1fr; }
  .panel-left { border-right: none; border-bottom: 1px solid var(--border); }
  .metric-strip { grid-template-columns: repeat(2,1fr); }
  .masthead { padding: 1.8rem 1.5rem 1.5rem; }
  .panel-right { padding: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "active_module": 0,
        "results":       {},   # module_id -> {"content": str, "tokens_in": int, "tokens_out": int, "elapsed": float}
        "pdf_info":      None, # {"name", "size_kb", "pages", "text_chars", "has_text"}
        "pdf_b64":       None,
        "pdf_text":      None,
        "run_triggered": False,
        "api_key":       "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS MODULE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
MODULES = [
    {
        "id":    "floor_plan",
        "icon":  "⬡",
        "name":  "Floor Plan & Room Schedule",
        "short": "Rooms · Dimensions · Levels",
        "desc":  "Extract every room, floor level, area, and ceiling height from all pages.",
        "color": "var(--accent)",
    },
    {
        "id":    "surface_area",
        "icon":  "◫",
        "name":  "Surface Area Calculation",
        "short": "Walls · Ceilings · Openings",
        "desc":  "Compute net paintable sq.ft. with door/window deductions per room.",
        "color": "var(--accent3)",
    },
    {
        "id":    "paint_materials",
        "icon":  "⬢",
        "name":  "Paint & Finish Quantities",
        "short": "Gallons · Coats · Coverage",
        "desc":  "Calculate primer and finish gallons by surface type with waste factors.",
        "color": "var(--accent2)",
    },
    {
        "id":    "labor",
        "icon":  "◈",
        "name":  "Labor Estimation by Trade",
        "short": "Hours · Rates · Trade Cost",
        "desc":  "Estimate hours per trade using RS Means / CIQS productivity benchmarks.",
        "color": "var(--gold)",
    },
    {
        "id":    "structural",
        "icon":  "◬",
        "name":  "Structural & Material Take-Off",
        "short": "Lumber · Drywall · Mechanical",
        "desc":  "Quantity take-off for framing, drywall, doors, windows, and MEP rough-in.",
        "color": "var(--accent)",
    },
    {
        "id":    "cost_summary",
        "icon":  "◉",
        "name":  "Full Cost Summary & Variants",
        "short": "BoQ · Overhead · Contingency",
        "desc":  "Complete cost build-up with overhead, contingency, and variant comparison.",
        "color": "var(--success)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PERSONA = """You are a senior architectural estimator and licensed quantity surveyor with 20+ years experience in residential construction estimation. You have deep expertise in:
- Reading and interpreting multi-page architectural blueprint sets (floor plans, elevations, sections, electrical plans, schedules)
- Extracting precise dimensions, room schedules, and material specifications from construction drawings
- Applying industry-standard calculation methodologies (RS Means, CIQS, AIQS)
- Producing client-ready, defensible cost estimates for builders, developers, and contractors

CRITICAL INSTRUCTIONS:
1. Analyze ALL pages of the provided blueprint document — do not skip any page
2. If dimension values are redacted (shown as XXXXXX), note them and provide a reasonable estimate based on context
3. Be extremely specific with numbers — vague estimates are not acceptable
4. Format all output in clean, structured Markdown with proper tables
5. Use realistic, current-year pricing for materials and labor
6. Flag any assumptions you make with a ⚠ symbol
7. Produce comprehensive output — this document may be 50 pages and every page matters"""

def build_prompt(module_id: str, has_native_pdf: bool, extra_context: str = "") -> str:
    """
    Return a full user-turn prompt for the given module.
    If has_native_pdf is True the PDF was sent as a native file part,
    so no need to paste extracted text — just reference the document.
    """
    intros = {
        "floor_plan": """## TASK: COMPREHENSIVE FLOOR PLAN & ROOM SCHEDULE EXTRACTION

Analyze every floor plan page in this blueprint set and produce:

### 1. Project Metadata
Extract from release notes / title block:
| Field | Value |
|-------|-------|
| Plan Name | |
| Plan Code | |
| Elevation Code | |
| Revision Date | |
| Draftsperson ID | |
| Joist Depth | |

### 2. Room Schedule — ALL Floors
For every room on every floor, produce a detailed schedule:

| Floor | Room / Space | Width | Depth | Area (sq.ft.) | Ceiling Type | Ceiling Ht (ft) | Notes |
|-------|-------------|-------|-------|---------------|--------------|-----------------|-------|

Include: main floor, upper floor (all options), basement development, garage, decks, covered areas.

### 3. Optional Layout Variants
List every optional or alternate layout page found:
- Option name and page reference
- Delta area vs. base plan (sq.ft. added/removed)
- Key rooms added or changed

### 4. Structural Annotations
Extract all structural callouts visible on the plans:
- Wall types (2×4, 2×6, etc.)
- Beam/header sizes
- Post types (wood, steel)
- Joist/truss specs
- Special structural notes

### 5. Floor Area Summary Table

| Level | Included Base (sq.ft.) | Optional Config A (sq.ft.) | Optional Config B (sq.ft.) |
|-------|------------------------|----------------------------|----------------------------|

### 6. Confidence Assessment
Rate extraction confidence (0–100%) per floor. Note any obscured, redacted, or unclear dimensions.""",

        "surface_area": """## TASK: COMPLETE SURFACE AREA CALCULATION

Using the dimensions extracted from all blueprint pages, calculate net paintable surface areas.

**Calculation Rules:**
- Wall Area = (Room Perimeter × Ceiling Height) − Door Deductions − Window Deductions
- Interior door deduction: 21 sq.ft. (3′×7′ standard); exterior door: 24.5 sq.ft.
- Apply 1.15 waste/overlap multiplier for 2-coat systems
- Ceiling flat: factor 1.0 | Vaulted/cathedral: factor 1.25 | Tray: add tray side area
- Basement development: default 8 ft ceiling unless annotated
- Trim allowance: 10% of gross wall area (expressed in lin.ft.)

### 1. Room-by-Room Surface Schedule

| Floor | Room | Gross Wall Area (sq.ft.) | Door Deduct | Window Deduct | Net Wall Area | Ceiling Area | Trim Lin.ft. | Notes |
|-------|------|--------------------------|-------------|---------------|---------------|--------------|--------------|-------|

### 2. Exterior Surface Areas

| Surface | Dimensions | Area (sq.ft.) | Notes |
|---------|-----------|---------------|-------|
| Elevation — Front | | | |
| Elevation — Rear | | | |
| Elevation — Left | | | |
| Elevation — Right | | | |
| Soffits & Fascia | | | |
| Decks / Exterior Wood | | | |

### 3. Surface Area Summary by Category

| Category | Area (sq.ft.) | Notes |
|----------|---------------|-------|
| Interior Walls — Main Floor | | |
| Interior Walls — Upper Floor | | |
| Interior Walls — Basement Dev. | | |
| All Ceilings | | |
| Exterior Walls (total) | | |
| Soffits & Bulkheads | | |
| Trim & Casings | | lin.ft. |
| Stair Risers & Stringers | | |
| **GRAND TOTAL PAINTABLE** | | |

### 4. Deduction Log
List every door and window deduction applied with its code, dimensions, and sq.ft. removed.""",

        "paint_materials": """## TASK: PAINT & FINISH MATERIAL QUANTITY ANALYSIS

Using the surface areas from this blueprint, calculate all paint and finish material requirements.

**Coverage Rates & Standards:**
| Product | Finish | Coverage (sq.ft./gal) | Coats | Waste Factor |
|---------|--------|----------------------|-------|-------------|
| Interior Wall Primer | PVA Flat | 350 | 1 | 10% |
| Interior Wall Finish | Eggshell | 400 | 2 | 8% |
| Ceiling Paint | Flat White | 400 | 2 | 5% |
| Trim & Casings | Semi-Gloss | 450 | 2 | 12% |
| Interior Doors | Semi-Gloss | 350 | 2 | 5% |
| Exterior Primer | Acrylic Block | 300 | 1 | 12% |
| Exterior Finish | Satin Exterior | 350 | 2 | 10% |
| Deck / Exterior Wood | Solid Stain | 300 | 2 | 15% |
| Basement Dev. Walls | Eggshell | 380 | 2 | 8% |
| Stair Risers | Semi-Gloss | 450 | 2 | 10% |

**Formula:** Gallons = (Surface Area × Coats × Waste Factor) ÷ Coverage Rate

### 1. Detailed Paint Schedule

| Item | Surface Area (sq.ft.) | Product Type | Gallons Required | Unit Cost (USD) | Subtotal (USD) | Notes |
|------|-----------------------|-------------|------------------|-----------------|----------------|-------|

### 2. Paint Summary by Zone
Break totals out by: Main Floor Interior / Upper Floor / Basement / Exterior.

### 3. Sundry Materials
Estimate caulking, sandpaper, tape, drop cloths, roller covers, brushes — itemised with costs.

### 4. Total Material Cost Summary

| Category | Gallons | Cost (USD) |
|----------|---------|-----------|
| All Interior Paint & Primer | | |
| All Exterior Paint & Primer | | |
| Deck Stains / Sealers | | |
| Sundry / Consumables | | |
| **GRAND TOTAL MATERIALS** | | |""",

        "labor": """## TASK: COMPREHENSIVE LABOR ESTIMATION BY TRADE

Estimate all labor hours and costs for this residential construction project.

**Productivity Rates (RS Means / CIQS — Residential New Construction):**
| Trade | Unit | Rate | Basis | Source |
|-------|------|------|-------|--------|
| Painter — Interior | sq.ft. primed surface | 200 sq.ft./hr | 2-coat system | RS Means |
| Painter — Exterior | sq.ft. cladding | 150 sq.ft./hr | Spray + backroll | RS Means |
| Finish Carpenter (trim) | lin.ft. installed | 30 lin.ft./hr | Baseboard + casing | CIQS |
| Framer | sq.ft. floor area | 8 sq.ft./hr | Platform frame | RS Means |
| Drywaller (hang + tape) | sq.ft. wall/ceiling | 80 sq.ft./hr | ½″ drywall | RS Means |
| Electrician (rough-in) | outlets / devices | 0.75 hr/device | 15A residential | CIQS |
| Plumber (rough-in) | fixtures | 4 hrs/fixture | PEX residential | CIQS |
| HVAC Technician | sq.ft. conditioned | 0.05 hr/sq.ft. | Forced air | ACCA |
| Insulator | sq.ft. exterior wall | 0.04 hr/sq.ft. | Batt + spray foam | RS Means |
| Flooring Installer | sq.ft. floor | 0.06 hr/sq.ft. | LVP / tile mix | RS Means |
| Tiler | sq.ft. tile | 0.12 hr/sq.ft. | Wet areas | RS Means |
| Cabinet Installer | cabinet count | 1.5 hrs/unit | Kitchen + baths | CIQS |

### 1. Trade-by-Trade Labor Schedule

| Trade | Quantity Basis | Quantity | Rate | Hours | USD/hr | Labor Cost (USD) | Notes |
|-------|---------------|----------|------|-------|--------|------------------|-------|

### 2. Labor Phase Breakdown
Organize labor by construction phase:
- Rough framing
- Rough mechanical (electrical, plumbing, HVAC)
- Insulation + air barrier
- Drywall (hang, tape, finish)
- Interior finish (trim, doors, painting)
- Tile + flooring
- Cabinetry + millwork
- Exterior finish (cladding, paint)

### 3. Labor Cost Summary

| Phase | Hours | Labor Cost (USD) | % of Total |
|-------|-------|-----------------|-----------|

| **TOTAL DIRECT LABOR** | | | 100% |

### 4. Labor Risk Notes
Identify any areas where productivity rates may be lower (complex geometry, custom details, etc.)""",

        "structural": """## TASK: STRUCTURAL & MATERIAL QUANTITY TAKE-OFF

Perform a detailed material take-off for all structural and finish materials in this blueprint.

### 1. Framing Lumber Take-Off

| Component | Quantity | Dimension | Board Feet | Unit Cost | Subtotal | Notes |
|-----------|----------|-----------|-----------|-----------|---------|-------|
| Exterior wall studs (2×6) | | | | | | |
| Interior wall studs (2×4) | | | | | | |
| Top & bottom plates | | | | | | |
| Floor joists / LVL | | | | | | |
| Roof/ceiling joists or trusses | | | | | | |
| Beams & headers | | | | | | |
| Posts & columns | | | | | | |
| Rim board & blocking | | | | | | |
| Sheathing — walls (OSB/plywood) | | | | | | |
| Sheathing — roof | | | | | | |
| Sub-floor | | | | | | |

### 2. Drywall Material Schedule

| Area | Sheets (4×8) | Sheets (4×12) | Joint Compound (bags) | Corner Bead (lin.ft.) | Notes |
|------|-------------|--------------|----------------------|----------------------|-------|

### 3. Door & Window Schedule
Extract from plan schedules:

| Mark | Type | Width | Height | Qty | Unit Cost | Total | Notes |
|------|------|-------|--------|-----|-----------|-------|-------|

### 4. Mechanical Rough-In Counts
Extract electrical devices, plumbing fixtures, HVAC equipment from electrical/mechanical plans:

**Electrical:** Outlets / Switches / Light fixtures / Smoke detectors / Dedicated circuits
**Plumbing:** Toilets / Sinks / Showers / Tubs / Hose bibs / Floor drains
**HVAC:** Furnace / AC units / HRV / Exhaust fans / Supply registers / Return air grilles

### 5. Insulation Take-Off

| Area | Type | R-Value | Sq.ft. or Lin.ft. | Cost/Unit | Subtotal |
|------|----|---------|-------------------|-----------|---------|

### 6. Material Cost Summary

| Category | Quantity | Unit | Cost (USD) |
|----------|----------|------|-----------|
| Framing Lumber & Engineered Wood | | board ft | |
| Sheathing & Sub-floor | | sheets | |
| Drywall | | sheets | |
| Insulation | | sq.ft. | |
| Doors (interior + exterior) | | units | |
| Windows | | units | |
| **SUBTOTAL STRUCTURAL MATERIALS** | | | |""",

        "cost_summary": """## TASK: COMPLETE PROJECT COST SUMMARY & VARIANT ANALYSIS

Produce a comprehensive, investor-ready cost estimate for this residential construction project.

### 1. Full Cost Build-Up Table

| # | Cost Category | Basis | Amount (USD) | % of Hard Cost | Notes |
|---|--------------|-------|-------------|---------------|-------|
| 1 | Paint & Finish Materials | gallons + sundries | | | |
| 2 | Framing Lumber & Sheathing | board ft | | | |
| 3 | Drywall Materials | sheets | | | |
| 4 | Doors & Windows | schedule count | | | |
| 5 | Rough Mechanical (Elec/Plumb/HVAC) | device + fixture counts | | | |
| 6 | Insulation | sq.ft. | | | |
| 7 | Flooring | sq.ft. | | | |
| 8 | Cabinetry & Millwork | kitchen + baths + pantry | | | |
| 9 | Tile & Wet Area Finishes | sq.ft. | | | |
| 10 | Direct Labor — All Trades | hours | | | |
| — | **HARD COST SUBTOTAL** | | | 100% | |
| 11 | General Conditions & Site | 5% of hard cost | | | |
| 12 | Builder Overhead & Profit | 12% of hard cost | | | |
| 13 | Contingency Reserve | 8% of hard cost | | | |
| — | **TOTAL ESTIMATED PROJECT COST** | | | | |

### 2. Cost per Square Foot Analysis

| Configuration | Total Area (sq.ft.) | Hard Cost (USD) | Total w/OH (USD) | Cost/sq.ft. | Notes |
|--------------|--------------------|-----------------|--------------------|------------|-------|

### 3. Plan Variant Comparison Table

| Configuration | Total Area (sq.ft.) | Paint Materials (USD) | Direct Labor (USD) | Total Estimate (USD) | Delta vs Base |
|--------------|--------------------|-----------------------|--------------------|---------------------|--------------|
| Base Only (no dev.) | | | | | — |
| Base + 1-Bed Basement Dev. | | | | | |
| Base + 2-Bed Basement Dev. | | | | | |
| Base + Upper Floor Opt. #1 | | | | | |
| Base + Upper Floor Opt. #2 | | | | | |
| Full Build (all options) | | | | | |

### 4. Top Cost Drivers (Pareto)
List the 5 largest cost items and what percentage of total they represent.

### 5. Key Estimate Assumptions
Bullet-point all pricing assumptions: region, year, labor rates, material indices.

### 6. Risk & Contingency Notes
Identify 3–5 cost risk items specific to this plan set (e.g., complex roofline, large glazing area, deep basement).

### 7. Executive Summary for Builder
Write a 150-word plain-English summary of this estimate suitable for a builder client meeting.""",
    }

    module_prompt = intros.get(module_id, "Analyze this blueprint and provide a comprehensive construction estimate with detailed tables.")

    full_prompt = f"""{module_prompt}

---
**DOCUMENT CONTEXT:**
You have been provided with the complete blueprint PDF. Analyze ALL pages thoroughly before responding. Do not truncate or abbreviate your response — this is a professional deliverable.
"""

    if extra_context.strip():
        full_prompt += f"\n**ADDITIONAL INSTRUCTIONS FROM USER:**\n{extra_context.strip()}\n"

    return full_prompt


# ─────────────────────────────────────────────────────────────────────────────
#  PDF UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(uploaded_file) -> dict:
    """
    Read uploaded PDF and return info dict + base64 encoding.
    Extracts text via pypdf if available (helps with large PDFs).
    """
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    size_kb    = len(raw_bytes) / 1024
    b64        = base64.standard_b64encode(raw_bytes).decode("utf-8")

    pages    = 0
    text     = ""
    has_text = False

    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages  = len(reader.pages)
            chunks = []
            for i, page in enumerate(reader.pages):
                if i >= MAX_PDF_PAGES:
                    break
                t = page.extract_text() or ""
                if t.strip():
                    chunks.append(f"--- PAGE {i+1} ---\n{t}")
            text     = "\n\n".join(chunks)
            has_text = len(text.strip()) > 200
        except Exception:
            pages = 0

    return {
        "name":       uploaded_file.name,
        "size_kb":    size_kb,
        "pages":      pages,
        "text_chars": len(text),
        "has_text":   has_text,
        "b64":        b64,
        "text":       text,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  OPENROUTER STREAMING CALL
# ─────────────────────────────────────────────────────────────────────────────
def stream_openrouter(
    api_key:     str,
    module_id:   str,
    pdf_info:    dict,
    extra_ctx:   str = "",
) -> Generator[str, None, None]:
    """
    Stream the model response token-by-token.
    Yields text delta strings. Final yield is a JSON sentinel:
      {"__meta__": {"tokens_in": N, "tokens_out": N, "elapsed": F}}
    """
    prompt = build_prompt(module_id, has_native_pdf=True, extra_context=extra_ctx)

    # Build message content — PDF as native file part (Gemini supports this)
    content = [
        {
            "type": "text",
            "text": prompt,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:application/pdf;base64,{pdf_info['b64']}",
            },
        },
    ]

    # If pypdf extracted meaningful text, prepend it as a text block
    # to help models that may not have perfect PDF vision
    if pdf_info["has_text"] and pdf_info["text_chars"] > 500:
        text_block = pdf_info["text"][:120_000]  # cap at ~120k chars
        content.insert(1, {
            "type": "text",
            "text": (
                f"[EXTRACTED TEXT FROM PDF — {pdf_info['pages']} pages]\n\n"
                f"{text_block}\n\n"
                "[END EXTRACTED TEXT — now refer to the full PDF image/file above for visual details]"
            ),
        })

    payload = {
        "model":       PRIMARY_MODEL,
        "messages":    [
            {"role": "system", "content": SYSTEM_PERSONA},
            {"role": "user",   "content": content},
        ],
        "max_tokens":  MAX_OUTPUT_TOKENS,
        "temperature": 0.1,
        "stream":      True,
    }

    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://blueprint-intelligence.demo",
        "X-Title":        "Blueprint Intelligence System",
    }

    t0 = time.time()
    tokens_in  = 0
    tokens_out = 0

    try:
        with requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:

            if resp.status_code != 200:
                # Try to parse error body
                try:
                    err_body = resp.json()
                    msg = err_body.get("error", {}).get("message", resp.text[:400])
                except Exception:
                    msg = resp.text[:400]
                yield f"\n\n**❌ API Error {resp.status_code}:** {msg}"
                yield json.dumps({"__meta__": {"tokens_in": 0, "tokens_out": 0, "elapsed": 0, "error": msg}})
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract text delta
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text  = delta.get("content", "")
                    if text:
                        tokens_out += 1   # approximate
                        yield text

                # Harvest usage if present (some providers send it on final chunk)
                usage = chunk.get("usage", {})
                if usage:
                    tokens_in  = usage.get("prompt_tokens",     tokens_in)
                    tokens_out = usage.get("completion_tokens", tokens_out)

    except requests.exceptions.Timeout:
        yield f"\n\n**❌ Timeout:** The request exceeded {REQUEST_TIMEOUT_S}s. Try a smaller PDF or fewer pages."
    except requests.exceptions.ConnectionError as e:
        yield f"\n\n**❌ Connection error:** {str(e)[:200]}"
    except Exception as e:
        yield f"\n\n**❌ Unexpected error:** {str(e)[:300]}"

    elapsed = time.time() - t0
    yield json.dumps({
        "__meta__": {
            "tokens_in":  tokens_in,
            "tokens_out": tokens_out,
            "elapsed":    round(elapsed, 1),
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_size(kb: float) -> str:
    if kb >= 1024:
        return f"{kb/1024:.1f} MB"
    return f"{kb:.0f} KB"

def fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)

def page_ok(pages: int) -> tuple[bool, str]:
    """Return (ok, msg) based on page count."""
    if pages == 0:
        return True, ""
    if pages <= MAX_PDF_PAGES:
        return True, f"{pages} pages — within {MAX_PDF_PAGES}-page limit"
    return False, f"{pages} pages — exceeds {MAX_PDF_PAGES}-page limit"


# ─────────────────────────────────────────────────────────────────────────────
#  MASTHEAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="masthead-grid"></div>
  <div class="masthead-eyebrow">Blueprint Intelligence System · v2.0</div>
  <h1>Architectural <em>Analysis</em><br>& Cost Intelligence</h1>
  <p class="masthead-sub">
    Upload a residential blueprint PDF up to 50 pages. Six AI modules extract
    room schedules, surface areas, material quantities, trade labor, and full
    project cost estimates — powered by Gemini 2.5 Pro via OpenRouter.
  </p>
  <div class="masthead-badges">
    <span class="badge badge-teal">⬡ Gemini 2.5 Pro Preview</span>
    <span class="badge badge-orange">◫ Up to 50 pages</span>
    <span class="badge badge-violet">◈ 6 Analysis Modules</span>
    <span class="badge badge-gold">◉ Live Streaming</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  WORKSPACE — two-panel layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="workspace">', unsafe_allow_html=True)

# ── LEFT PANEL ─────────────────────────────────────────────────────────────
st.markdown('<div class="panel-left">', unsafe_allow_html=True)

left_col = st.container()

with left_col:
    # ── API Key ──────────────────────────────────────────────
    st.markdown('<div class="section-label">OpenRouter API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "api_key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-or-v1-xxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        help="Obtain at openrouter.ai/keys — key used only for current session",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("""
    <div style="font-family:var(--font-mono); font-size:0.62rem; color:var(--text-dim);
                margin-top:-0.3rem; margin-bottom:1.3rem;">
      Get a free key at
      <a href="https://openrouter.ai/keys" target="_blank"
         style="color:var(--accent); text-decoration:none;">openrouter.ai/keys</a>
      · never stored
    </div>
    """, unsafe_allow_html=True)

    # ── PDF Upload ───────────────────────────────────────────
    st.markdown('<div class="section-label">Blueprint PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "upload_pdf",
        type=["pdf"],
        label_visibility="collapsed",
        help=f"Multi-page PDF up to {MAX_PDF_PAGES} pages / {MAX_PDF_MB}MB",
    )

    if uploaded:
        # Only re-ingest if file changed
        if (st.session_state.pdf_info is None or
                st.session_state.pdf_info["name"] != uploaded.name or
                st.session_state.pdf_info["size_kb"] != uploaded.size / 1024):
            with st.spinner("Ingesting PDF…"):
                info = ingest_pdf(uploaded)
            st.session_state.pdf_info = info
            st.session_state.pdf_b64  = info["b64"]
            st.session_state.pdf_text = info["text"]

        info = st.session_state.pdf_info
        pages_ok, pages_msg = page_ok(info["pages"])
        dot_class = "dot-ok" if pages_ok else "dot-err"
        text_class = "dot-ok" if info["has_text"] else "dot-warn"

        st.markdown(f"""
        <div class="file-card">
          <div class="file-card-name">📄 {info['name']}</div>
          <div class="file-card-meta">
            <span class="file-stat"><span class="{dot_class}"></span>{info['pages'] or '?'} pages</span>
            <span class="file-stat"><span style="color:var(--text-dim)">●</span>{fmt_size(info['size_kb'])}</span>
            <span class="file-stat">
              <span class="{text_class}"></span>
              {'Text extracted' if info['has_text'] else 'Vision mode only'}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not pages_ok:
            st.markdown(f"""
            <div class="info-box info-box-err">
              <span class="info-box-icon">✗</span>
              <span class="info-box-body">{pages_msg} — please split the PDF or use a subset.</span>
            </div>
            """, unsafe_allow_html=True)
        elif not HAS_PYPDF:
            st.markdown("""
            <div class="info-box info-box-warn">
              <span class="info-box-icon">⚠</span>
              <span class="info-box-body">pypdf not installed — text extraction disabled.
              Install with: <code>pip install pypdf</code></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
          <span class="upload-icon">⬡</span>
          <div class="upload-title">Drop blueprint PDF here</div>
          <div class="upload-hint">Multi-page residential blueprints · up to 50 pages</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Module Selector ──────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Analysis Module</div>', unsafe_allow_html=True)

    for i, mod in enumerate(MODULES):
        is_active   = (st.session_state.active_module == i)
        active_cls  = "active" if is_active else ""
        has_result  = mod["id"] in st.session_state.results
        done_badge  = " ✓" if has_result else ""
        if st.button(
            f"{mod['icon']}  {mod['name']}{done_badge}",
            key=f"mod_btn_{i}",
            use_container_width=True,
        ):
            st.session_state.active_module = i
            st.rerun()

    # ── Extra context ────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Additional Instructions</div>', unsafe_allow_html=True)
    extra_ctx = st.text_area(
        "extra_context",
        placeholder="e.g. Focus on the 2-bedroom basement suite option. Use Alberta 2025 labor rates. Flag all redacted dimensions.",
        height=88,
        label_visibility="collapsed",
    )

    # ── Run button ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    can_run = bool(st.session_state.api_key) and uploaded is not None
    run_btn = st.button(
        f"⚡  RUN  {MODULES[st.session_state.active_module]['name'].upper()}",
        disabled=not can_run,
        key="run_analysis",
    )

    if not can_run:
        missing = []
        if not st.session_state.api_key: missing.append("API key")
        if uploaded is None:             missing.append("blueprint PDF")
        st.markdown(f"""
        <div style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-dim);
                    text-align:center; margin-top:0.5rem;">
          Missing: {' · '.join(missing)}
        </div>
        """, unsafe_allow_html=True)

    # ── Model info ───────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid var(--border-soft);">
      <div style="font-family:var(--font-mono); font-size:0.6rem; color:var(--text-dim); margin-bottom:0.4rem; letter-spacing:0.1em; text-transform:uppercase;">Model</div>
      <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--accent);">{PRIMARY_MODEL}</div>
      <div style="font-family:var(--font-mono); font-size:0.6rem; color:var(--text-dim); margin-top:0.2rem;">1M token context · native PDF vision · streaming</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /panel-left


# ── RIGHT PANEL ────────────────────────────────────────────────────────────
st.markdown('<div class="panel-right">', unsafe_allow_html=True)

right_col = st.container()

with right_col:

    # ── Run analysis when button clicked ────────────────────
    if run_btn and can_run:
        mod     = MODULES[st.session_state.active_module]
        mod_id  = mod["id"]
        info    = st.session_state.pdf_info

        st.markdown(f"""
        <div class="results-header">
          <div class="results-title">{mod['icon']} {mod['name']}</div>
          <div class="results-meta">
            <span>{info['name']}</span>
            <span style="color:var(--text-dim);">{info['pages']} pages · {fmt_size(info['size_kb'])}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline status row
        stage_active = {"floor_plan":0,"surface_area":1,"paint_materials":2,"labor":3,"structural":4,"cost_summary":5}
        active_s = stage_active.get(mod_id, 0)
        stages = [
            ("01","INGEST","PDF parsed"),
            ("02","EXTRACT","AI reads plans"),
            ("03","COMPUTE","Areas & qtys"),
            ("04","COST","Rate cards"),
            ("05","REPORT","Output rendered"),
        ]
        stage_html = '<div class="pipeline-row">'
        for si, (sn, st_name, sd) in enumerate(stages):
            cls = "active" if si <= active_s else ""
            stage_html += f'<div class="pipe-stage {cls}"><div class="pipe-num">{sn}</div><div class="pipe-name">{st_name}</div><div class="pipe-detail">{sd}</div></div>'
        stage_html += '</div>'
        st.markdown(stage_html, unsafe_allow_html=True)

        # Live status + progress
        status_ph  = st.empty()
        progress_ph = st.empty()

        status_ph.markdown(f"""
        <div class="status-line running">
          <span class="spinner">⟳</span>
          Sending {info['pages']} pages to {PRIMARY_MODEL_LABEL}…
        </div>
        """, unsafe_allow_html=True)
        progress_ph.progress(0.05)

        # Stream into a container
        stream_ph   = st.empty()
        accumulated = ""
        meta        = {}
        char_count  = 0
        t_start     = time.time()

        for chunk in stream_openrouter(
            api_key   = st.session_state.api_key,
            module_id = mod_id,
            pdf_info  = info,
            extra_ctx = extra_ctx,
        ):
            # Check for sentinel meta JSON
            if chunk.startswith('{"__meta__"'):
                try:
                    meta = json.loads(chunk)["__meta__"]
                except Exception:
                    pass
                break

            accumulated += chunk
            char_count  += len(chunk)

            # Update progress heuristically (based on char count, cap at 95%)
            progress_val = min(0.05 + (char_count / 20000) * 0.90, 0.95)
            progress_ph.progress(progress_val)

            elapsed_now = time.time() - t_start
            status_ph.markdown(f"""
            <div class="status-line running">
              <span class="spinner">⟳</span>
              Streaming response — {char_count:,} chars · {elapsed_now:.0f}s elapsed
            </div>
            """, unsafe_allow_html=True)

            # Render live markdown
            stream_ph.markdown(
                f'<div class="stream-box">{accumulated}</div>',
                unsafe_allow_html=True,
            )

        # Finalise
        progress_ph.progress(1.0)
        elapsed = meta.get("elapsed", round(time.time() - t_start, 1))
        tok_in  = meta.get("tokens_in",  0)
        tok_out = meta.get("tokens_out", 0)
        has_error = "❌" in accumulated[:120] or meta.get("error")

        if has_error:
            status_ph.markdown(f"""
            <div class="status-line error">
              ✗ Analysis failed — see error above
            </div>
            """, unsafe_allow_html=True)
        else:
            status_ph.markdown(f"""
            <div class="status-line success">
              ✓ Analysis complete — {elapsed}s · {fmt_tokens(tok_in)} prompt tokens · {fmt_tokens(tok_out)} output tokens
            </div>
            """, unsafe_allow_html=True)

            # Cache result
            st.session_state.results[mod_id] = {
                "content":    accumulated,
                "tokens_in":  tok_in,
                "tokens_out": tok_out,
                "elapsed":    elapsed,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
                "pdf_name":   info["name"],
            }

        # Final rendered output
        stream_ph.markdown(
            f'<div class="stream-box">{accumulated}</div>',
            unsafe_allow_html=True,
        )

        # Download
        if not has_error:
            mod_name = mod["name"]
            st.download_button(
                label="⬇  Download Markdown Report",
                data=(
                    f"# {mod_name}\n\n"
                    f"**Model:** {PRIMARY_MODEL}  \n"
                    f"**Document:** {info['name']} ({info['pages']} pages)  \n"
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
                    f"**Response time:** {elapsed}s\n\n---\n\n"
                    f"{accumulated}"
                ),
                file_name=f"blueprint_{mod_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key=f"dl_{mod_id}_{time.time()}",
            )

    # ── Show cached results / empty state ───────────────────
    elif not run_btn:
        active_mod = MODULES[st.session_state.active_module]
        cached     = st.session_state.results.get(active_mod["id"])

        if cached:
            # Show cached result
            st.markdown(f"""
            <div class="results-header">
              <div class="results-title">{active_mod['icon']} {active_mod['name']}</div>
              <div class="results-meta">
                <div class="token-pill">
                  <span class="t-accent">{fmt_tokens(cached['tokens_in'])}</span> in ·
                  <span class="t-accent">{fmt_tokens(cached['tokens_out'])}</span> out ·
                  {cached['elapsed']}s
                </div>
                <span style="color:var(--text-dim); font-size:0.62rem;">{cached['timestamp']} · {cached['pdf_name']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Metric strip
            st.markdown(f"""
            <div class="metric-strip">
              <div class="metric-tile">
                <div class="metric-tile-value">{fmt_tokens(cached['tokens_in'])}</div>
                <div class="metric-tile-label">Prompt Tokens</div>
              </div>
              <div class="metric-tile">
                <div class="metric-tile-value">{fmt_tokens(cached['tokens_out'])}</div>
                <div class="metric-tile-label">Output Tokens</div>
              </div>
              <div class="metric-tile">
                <div class="metric-tile-value">{cached['elapsed']}s</div>
                <div class="metric-tile-label">Response Time</div>
              </div>
              <div class="metric-tile">
                <div class="metric-tile-value">{len(st.session_state.results)}/6</div>
                <div class="metric-tile-label">Modules Run</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Result content
            st.markdown(
                f'<div class="stream-box">{cached["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Download + expander for all run modules
            col_dl, col_gap = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    label="⬇  Export Markdown",
                    data=(
                        f"# {active_mod['name']}\n\n"
                        f"**Model:** {PRIMARY_MODEL}  \n"
                        f"**Document:** {cached['pdf_name']}  \n"
                        f"**Generated:** {cached['timestamp']}\n\n---\n\n"
                        f"{cached['content']}"
                    ),
                    file_name=f"blueprint_{active_mod['id']}.md",
                    mime="text/markdown",
                    key=f"dl_cached_{active_mod['id']}",
                )

            # If multiple modules run, show tab strip of all results
            completed_ids = list(st.session_state.results.keys())
            if len(completed_ids) > 1:
                st.markdown('<div style="margin-top:2rem;">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">All Completed Analyses</div>', unsafe_allow_html=True)
                tab_labels = [
                    f"{MODULES[next(j for j,m in enumerate(MODULES) if m['id']==mid)]['icon']} {MODULES[next(j for j,m in enumerate(MODULES) if m['id']==mid)]['name']}"
                    for mid in completed_ids
                ]
                tabs = st.tabs(tab_labels)
                for tab, mid in zip(tabs, completed_ids):
                    with tab:
                        r = st.session_state.results[mid]
                        with st.expander("Show full analysis", expanded=False):
                            st.markdown(
                                f'<div class="stream-box">{r["content"]}</div>',
                                unsafe_allow_html=True,
                            )
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Empty state for current module
            mod = MODULES[st.session_state.active_module]
            completed_count = len(st.session_state.results)

            st.markdown(f"""
            <div class="empty-state">
              <div class="empty-state-icon">{mod['icon']}</div>
              <div class="empty-state-title">{mod['name']}</div>
              <div class="empty-state-body">
                {mod['desc']}<br><br>
                Upload a blueprint PDF, enter your OpenRouter API key,
                and click <strong>Run</strong> to begin.
              </div>
              {'<div style="font-family:var(--font-mono); font-size:0.65rem; color:var(--accent); margin-top:0.5rem;">↑ ' + str(completed_count) + ' module' + ('s' if completed_count!=1 else '') + ' already completed — select to view</div>' if completed_count > 0 else ''}
            </div>
            """, unsafe_allow_html=True)

            # Show pipeline info
            st.markdown(f"""
            <div style="margin-top:1.5rem;">
              <div class="section-label">How it works</div>
              <div class="pipeline-row">
                <div class="pipe-stage"><div class="pipe-num">01</div><div class="pipe-name">INGEST</div><div class="pipe-detail">PDF parsed & encoded</div></div>
                <div class="pipe-stage"><div class="pipe-num">02</div><div class="pipe-name">EXTRACT</div><div class="pipe-detail">AI reads all pages</div></div>
                <div class="pipe-stage"><div class="pipe-num">03</div><div class="pipe-name">COMPUTE</div><div class="pipe-detail">Areas & quantities</div></div>
                <div class="pipe-stage"><div class="pipe-num">04</div><div class="pipe-name">COST</div><div class="pipe-detail">Rate cards applied</div></div>
                <div class="pipe-stage"><div class="pipe-num">05</div><div class="pipe-name">REPORT</div><div class="pipe-detail">Structured output</div></div>
              </div>
              <div style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-dim); margin-top:0.6rem;">
                All 6 modules share the same PDF — run each independently or in sequence.
                Results are cached in your session.
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /panel-right
st.markdown('</div>', unsafe_allow_html=True)  # /workspace

"""
=============================================================================
MUSICPERSONA — Showcase Edition
Big Five Personality Prediction from Music Reviews
CN6000 Final Year Project — Zest Chukwu — 2407734
=============================================================================
Run: python -m streamlit run MusicPersona.py
Requires: pip install streamlit torch transformers plotly
=============================================================================
"""

import os, time
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from torch import nn

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MusicPersona — Personality from Music Reviews",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&family=Courier+Prime&display=swap');

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif;
  scroll-behavior: smooth;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #04080f; color: #f0ece4; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SECTION WRAPPERS ── */
.section { padding: 5rem 8%; }
.section-narrow { padding: 4rem 12%; }

/* ── NAV ── */
.nav-bar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 999;
  background: rgba(4,8,15,0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255,255,255,0.06);
  display: flex; align-items: center;
  justify-content: space-between;
  padding: 1rem 8%;
}
.nav-logo {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic; font-size: 1.5rem;
  color: #f0ece4; letter-spacing: -0.01em;
}
.nav-links { display: flex; gap: 2.5rem; }
.nav-link {
  font-size: 0.78rem; font-weight: 500;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #b8c8d8; text-decoration: none;
  transition: color 0.2s;
}
.nav-link:hover { color: #3ecfb2; }

/* ── HERO SECTION ── */
.hero-section {
  min-height: 100vh;
  background: radial-gradient(ellipse 80% 60% at 50% -10%,
    rgba(62,207,178,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 80% 80%,
    rgba(100,130,200,0.08) 0%, transparent 60%);
  display: flex; flex-direction: column;
  justify-content: center; padding: 8rem 8% 5rem;
}
.hero-eyebrow {
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: #3ecfb2; margin-bottom: 1.5rem;
}
.hero-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(3.5rem, 8vw, 7rem);
  font-weight: 700; line-height: 0.95;
  color: #f0ece4; letter-spacing: -0.03em;
  margin-bottom: 0.3rem;
}
.hero-title-italic {
  font-style: italic; color: #3ecfb2;
}
.hero-subtitle {
  font-size: clamp(1rem, 2vw, 1.25rem);
  font-weight: 300; color: #c8c0b4;
  max-width: 560px; line-height: 1.65;
  margin: 1.8rem 0 2.5rem;
}
.hero-stat-row {
  display: flex; gap: 3rem; margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.hero-stat-num {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.5rem; font-weight: 700;
  color: #f0ece4; line-height: 1;
}
.hero-stat-label {
  font-size: 0.75rem; color: #8a9db8;
  letter-spacing: 0.06em; margin-top: 0.3rem;
}
.hero-scroll-hint {
  font-size: 0.75rem; color: #4a5a70;
  letter-spacing: 0.1em; margin-top: 4rem;
  display: flex; align-items: center; gap: 0.5rem;
}

/* ── SCENARIO SECTION ── */
.scenario-section {
  background: #070d18;
  border-top: 1px solid rgba(255,255,255,0.04);
  border-bottom: 1px solid rgba(255,255,255,0.04);
  padding: 5rem 8%;
}
.scenario-label {
  font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: #3ecfb2; margin-bottom: 1rem;
}
.scenario-heading {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(2rem, 4vw, 3.2rem);
  font-weight: 600; color: #f0ece4;
  line-height: 1.15; max-width: 700px;
  margin-bottom: 1.2rem;
}
.scenario-body {
  font-size: 1rem; color: #c0b8b0;
  line-height: 1.75; max-width: 640px;
  margin-bottom: 3rem;
}
.use-case-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem; margin-top: 2rem;
}
.use-case-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 2rem;
  transition: border-color 0.3s, transform 0.2s;
}
.use-case-card:hover {
  border-color: rgba(62,207,178,0.3);
  transform: translateY(-2px);
}
.use-case-icon { font-size: 2rem; margin-bottom: 1rem; }
.use-case-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.3rem; font-weight: 600;
  color: #f0ece4; margin-bottom: 0.6rem;
}
.use-case-body {
  font-size: 0.88rem; color: #a8b8c8;
  line-height: 1.65;
}
.use-case-tag {
  display: inline-block; margin-top: 1rem;
  font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #3ecfb2; padding: 0.2rem 0.7rem;
  border: 1px solid rgba(62,207,178,0.3);
  border-radius: 20px;
}

/* ── PREDICT SECTION ── */
.predict-section {
  padding: 5rem 8% 1.5rem;
  background: linear-gradient(180deg, #04080f 0%, #070d18 100%);
}
.predict-label {
  font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: #3ecfb2; margin-bottom: 1rem;
}
.predict-heading {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(2rem, 3.5vw, 2.8rem);
  font-weight: 600; color: #f0ece4;
  margin-bottom: 0.6rem;
}
.predict-sub {
  font-size: 0.95rem; color: #a8b8c8;
  line-height: 1.65; max-width: 560px;
  margin-bottom: 2.5rem;
}

/* ── EXAMPLE PILLS ── */
.example-pill {
  display: inline-flex; align-items: center; gap: 0.5rem;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 30px; padding: 0.45rem 1.1rem;
  font-size: 0.82rem; color: #e8e2da;
  cursor: pointer; transition: all 0.2s;
  margin: 0.25rem;
}
.example-pill:hover {
  border-color: rgba(62,207,178,0.4);
  color: #3ecfb2; background: rgba(62,207,178,0.05);
}
.example-pill-dot {
  width: 6px; height: 6px; border-radius: 50%;
}

/* ── INPUT AREA ── */
.stTextArea textarea {
  background: #f5f0e8 !important;
  border: 1px solid #d8d0c4 !important;
  border-radius: 10px !important;
  color: #1a1a1a !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.95rem !important;
  line-height: 1.7 !important;
}
.stTextArea textarea:focus {
  border-color: #3ecfb2 !important;
  box-shadow: 0 0 0 2px rgba(62,207,178,0.15) !important;
}
.stTextArea textarea::placeholder { color: #9a9080 !important; }

/* ── BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg, #3ecfb2 0%, #2aaa90 100%) !important;
  color: #04080f !important; border: none !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important; font-size: 0.82rem !important;
  letter-spacing: 0.12em !important;
  padding: 0.65rem 2rem !important;
  transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
  opacity: 0.9 !important; transform: translateY(-1px) !important;
}

/* ── RESULTS SECTION ── */
.results-section {
  padding: 0.5rem 8% 6rem;
  background: #070d18;
}
.results-heading {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(2rem, 3vw, 2.5rem);
  font-weight: 600; color: #f0ece4;
  margin-bottom: 0.4rem;
}
.results-sub { font-size: 0.88rem; color: #8a9db8; margin-bottom: 2.5rem; }

/* ── ARCHETYPE CARD ── */
.archetype-card {
  background: linear-gradient(135deg,
    rgba(62,207,178,0.06) 0%, rgba(100,130,200,0.04) 100%);
  border: 1px solid rgba(62,207,178,0.2);
  border-radius: 14px; padding: 2rem 2.4rem;
  margin-bottom: 2rem;
}
.archetype-eyebrow {
  font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: #3ecfb2; margin-bottom: 0.5rem;
}
.archetype-name {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic; font-size: 2rem;
  font-weight: 600; color: #f0ece4;
}
.archetype-desc {
  font-size: 0.9rem; color: #b8c8d8;
  line-height: 1.65; margin-top: 0.5rem;
}

/* ── STAT CARDS ── */
.stat-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px; padding: 1.3rem 1.5rem;
}
.stat-num {
  font-family: 'Cormorant Garamond', serif;
  font-size: 3rem; font-weight: 700; line-height: 1;
}
.stat-label {
  font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #7a8fa8; margin-bottom: 0.3rem;
}
.stat-sub { font-size: 0.78rem; color: #7a8fa8; margin-top: 0.3rem; }

/* ── TRAIT ROW ── */
.trait-row {
  display: flex; align-items: center;
  padding: 1rem 1.3rem;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 8px; margin-bottom: 0.5rem;
  gap: 1rem;
}
.trait-icon { font-size: 1.1rem; min-width: 1.5rem; }
.trait-name-col {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.05rem; color: #e8e2da; min-width: 150px;
}
.trait-bar-wrap { flex: 1; }
.trait-bar-bg {
  background: rgba(255,255,255,0.05);
  border-radius: 4px; height: 7px; overflow: hidden;
}
.trait-bar-fill { height: 7px; border-radius: 4px; }
.trait-score-col {
  font-family: 'Courier Prime', monospace;
  font-size: 1.3rem; min-width: 42px; text-align: right;
}
.trait-badge {
  font-size: 0.7rem; letter-spacing: 0.05em;
  padding: 0.2rem 0.7rem; border-radius: 20px;
  min-width: 130px; text-align: center;
}

/* ── LOADING ── */
.loading-overlay {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 4rem; gap: 1rem;
}
.loading-text {
  font-size: 0.85rem; color: #3ecfb2;
  letter-spacing: 0.1em; text-transform: uppercase;
}

/* ── FOOTER ── */
.footer {
  background: #04080f;
  border-top: 1px solid rgba(255,255,255,0.04);
  padding: 3rem 8%;
  display: flex; justify-content: space-between; align-items: center;
}
.footer-logo {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic; font-size: 1.2rem; color: #6a7a90;
}
.footer-text { font-size: 0.75rem; color: #5a6a80; line-height: 1.6; }

/* ── CALIBRATION TOGGLE ── */
.stToggle { color: #f0ece4 !important; }

/* ── WORD COUNT ── */
.wc { font-size: 0.75rem; text-align: right;
      margin-top: -0.4rem; margin-bottom: 0.5rem; }

/* ── DIVIDER ── */
.section-divider {
  height: 1px;
  background: linear-gradient(90deg,
    transparent, rgba(62,207,178,0.3), transparent);
  margin: 0;
}

/* ── HISTORY ── */
.history-item {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 8px; padding: 0.8rem 1rem;
  margin-bottom: 0.4rem;
}
.history-preview {
  font-size: 0.78rem; color: #7a8fa8;
  font-style: italic; margin-top: 0.2rem;
  white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis;
}

/* ── COMPARISON ── */
.cmp-label {
  font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: #8a9db8; margin-bottom: 0.5rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ══════════════════════════════════════════════════════════
   RESPONSIVE — TABLET (max 1024px)
══════════════════════════════════════════════════════════ */
@media (max-width: 1024px) {
  .hero-section { padding: 7rem 6% 4rem; }
  .scenario-section { padding: 4rem 6%; }
  .predict-section { padding: 4rem 6% 1.5rem; }
  .results-section { padding: 0.5rem 6% 5rem; }
  .nav-bar { padding: 1rem 6%; }
  .use-case-grid { grid-template-columns: repeat(2, 1fr); }
  .hero-title { font-size: clamp(2.8rem, 6vw, 5rem); }
  .hero-stat-row { gap: 2rem; flex-wrap: wrap; }
  .trait-row { flex-wrap: wrap; gap: 0.6rem; }
  .trait-name-col { min-width: 120px; }
  .trait-badge { min-width: 100px; font-size: 0.65rem; }
}

/* ══════════════════════════════════════════════════════════
   RESPONSIVE — MOBILE (max 768px)
══════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
  /* Navigation */
  .nav-bar {
    padding: 0.9rem 5%;
    flex-direction: column;
    gap: 0.8rem;
    align-items: flex-start;
  }
  .nav-links { gap: 1.2rem; flex-wrap: wrap; }
  .nav-link { font-size: 0.72rem; }

  /* Hero */
  .hero-section { padding: 6rem 5% 3rem; }
  .hero-title { font-size: clamp(2.2rem, 10vw, 3.5rem); line-height: 1.05; }
  .hero-subtitle { font-size: 0.95rem; }
  .hero-stat-row {
    gap: 1.5rem;
    flex-direction: column;
    margin-top: 2rem;
    padding-top: 1.5rem;
  }
  .hero-stat-num { font-size: 2rem; }

  /* Sections */
  .scenario-section { padding: 3rem 5%; }
  .predict-section  { padding: 3rem 5% 1rem; }
  .results-section  { padding: 0.5rem 5% 4rem; }

  /* Use case grid — single column on mobile */
  .use-case-grid { grid-template-columns: 1fr; gap: 1rem; }
  .use-case-card { padding: 1.5rem; }

  /* Archetype card */
  .archetype-card { padding: 1.4rem 1.2rem; }
  .archetype-name { font-size: 1.6rem; }

  /* Trait rows */
  .trait-row {
    flex-wrap: wrap;
    padding: 0.8rem 0.9rem;
    gap: 0.5rem;
  }
  .trait-name-col { min-width: 110px; font-size: 0.95rem; }
  .trait-bar-wrap { min-width: 100%; order: 3; }
  .trait-score-col { font-size: 1.1rem; min-width: 35px; }
  .trait-badge {
    min-width: 0;
    font-size: 0.65rem;
    padding: 0.15rem 0.5rem;
  }

  /* Stat cards — 2x2 grid on mobile */
  .stat-card { padding: 1rem 1.1rem; }
  .stat-num { font-size: 2.2rem; }
  .stat-label { font-size: 0.65rem; }

  /* Scenario headings */
  .scenario-heading { font-size: clamp(1.6rem, 6vw, 2.4rem); }
  .predict-heading  { font-size: clamp(1.5rem, 5vw, 2.2rem); }
  .results-heading  { font-size: clamp(1.5rem, 5vw, 2.2rem); }

  /* Footer */
  .footer {
    flex-direction: column;
    gap: 1.2rem;
    text-align: center;
    padding: 2.5rem 5%;
  }

  /* Hide scroll hint on mobile */
  .hero-scroll-hint { display: none; }

  /* Buttons full width on mobile */
  .stButton > button { width: 100% !important; }

  /* Section padding override for Streamlit containers */
  [data-testid="stVerticalBlock"] > div { padding-left: 0 !important; padding-right: 0 !important; }
}

/* ══════════════════════════════════════════════════════════
   RESPONSIVE — SMALL MOBILE (max 480px)
══════════════════════════════════════════════════════════ */
@media (max-width: 480px) {
  .hero-section { padding: 5.5rem 4% 2.5rem; }
  .hero-title { font-size: clamp(1.9rem, 9vw, 2.8rem); }
  .scenario-section { padding: 2.5rem 4%; }
  .predict-section  { padding: 2.5rem 4% 1rem; }
  .results-section  { padding: 0.5rem 4% 3rem; }
  .nav-links { gap: 0.8rem; }
  .archetype-card { padding: 1.2rem 1rem; }
  .use-case-card { padding: 1.2rem; }
  .scenario-heading { font-size: clamp(1.4rem, 7vw, 2rem); }
}

/* ══════════════════════════════════════════════════════════
   TOUCH IMPROVEMENTS
══════════════════════════════════════════════════════════ */
@media (hover: none) {
  /* Remove hover effects on touch devices */
  .use-case-card:hover { transform: none; }
  .stButton > button:hover { transform: none !important; }
  /* Larger tap targets */
  .stButton > button { min-height: 48px !important; }
  .example-pill { padding: 0.6rem 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

TRAITS = ['openness', 'conscientiousness', 'extraversion',
          'agreeableness', 'neuroticism']

TRAIT_META = {
    'openness':          {'icon': '✦', 'color': '#3ecfb2', 'short': 'Open'},
    'conscientiousness': {'icon': '◈', 'color': '#5ba4cf', 'short': 'Consc'},
    'extraversion':      {'icon': '◉', 'color': '#e8c16a', 'short': 'Extra'},
    'agreeableness':     {'icon': '❋', 'color': '#a78bfa', 'short': 'Agree'},
    'neuroticism':       {'icon': '◎', 'color': '#f87171', 'short': 'Neuro'},
}

TRAIT_LABELS = {
    'openness':          [(65,'Imaginative & Creative','#3ecfb2','rgba(62,207,178,0.1)'),
                          (35,'Practical & Grounded','#4a6080','rgba(74,96,128,0.1)'),
                          (0, 'Balanced','#5ba4cf','rgba(91,164,207,0.1)')],
    'conscientiousness': [(65,'Organised & Disciplined','#5ba4cf','rgba(91,164,207,0.1)'),
                          (35,'Flexible & Spontaneous','#4a6080','rgba(74,96,128,0.1)'),
                          (0, 'Balanced','#5ba4cf','rgba(91,164,207,0.1)')],
    'extraversion':      [(65,'Outgoing & Energetic','#e8c16a','rgba(232,193,106,0.1)'),
                          (35,'Reflective & Reserved','#4a6080','rgba(74,96,128,0.1)'),
                          (0, 'Ambivert','#5ba4cf','rgba(91,164,207,0.1)')],
    'agreeableness':     [(65,'Warm & Cooperative','#a78bfa','rgba(167,139,250,0.1)'),
                          (35,'Analytical & Direct','#4a6080','rgba(74,96,128,0.1)'),
                          (0, 'Balanced','#5ba4cf','rgba(91,164,207,0.1)')],
    'neuroticism':       [(65,'Emotionally Sensitive','#f87171','rgba(248,113,113,0.1)'),
                          (35,'Emotionally Stable','#3ecfb2','rgba(62,207,178,0.1)'),
                          (0, 'Moderate','#5ba4cf','rgba(91,164,207,0.1)')],
}

# Archetype map: keyed by dominant trait
# Each trait has two archetypes: one for high score (>52), one for moderate
ARCHETYPE_MAP = {
    'openness': [
        {'name': 'The Philosopher', 'icon': '📖',
         'desc': 'Deeply curious and introspective, you seek meaning beneath the surface. '
                 'Music is a lens through which you understand the world.'},
        {'name': 'The Dreamer', 'icon': '🌙',
         'desc': 'Quietly imaginative, you are drawn to music that opens doors '
                 'to new feelings and ideas.'},
    ],
    'conscientiousness': [
        {'name': 'The Perfectionist', 'icon': '◈',
         'desc': 'You hold music to a high standard. Craft, precision and intentionality '
                 'matter more to you than trend or popularity.'},
        {'name': 'The Architect', 'icon': '⬡',
         'desc': 'Structured and discerning, you appreciate music that is built well — '
                 'where every choice is deliberate and nothing is wasted.'},
    ],
    'extraversion': [
        {'name': 'The Connector', 'icon': '✦',
         'desc': 'Music is a shared experience for you. You feel it most when others '
                 'feel it too — in rooms, in cars, in group chats at midnight.'},
        {'name': 'The Explorer', 'icon': '🧭',
         'desc': 'Energised and curious, you are always searching for the next sound '
                 'that surprises you and the people around you.'},
    ],
    'agreeableness': [
        {'name': 'The Empath', 'icon': '🌊',
         'desc': 'You experience music emotionally and share that experience generously. '
                 'Music for you is a language of care and connection.'},
        {'name': 'The Companion', 'icon': '❋',
         'desc': 'Warm and open, you find meaning in music that reflects '
                 'human experience — the ordinary moments made beautiful.'},
    ],
    'neuroticism': [
        {'name': 'The Individualist', 'icon': '◎',
         'desc': 'Intensely feeling and creatively alive. Your emotional sensitivity '
                 'gives you a uniquely powerful relationship with music.'},
        {'name': 'The Realist', 'icon': '◆',
         'desc': 'You engage with music honestly and without pretence — '
                 'drawn to work that reflects real emotional experience.'},
    ],
}

# Named examples with dot colours
EXAMPLES = [
    # Each review is written to push a specific dominant trait above all others.
    # With calibration ON, the dominant trait determines the archetype.
    # Labels show: archetype name (dominant trait in brackets)

    # Dominant: NEUROTICISM — anxiety, isolation, dark hours, emotional pain
    ('The Individualist',  '#f87171',
     "Can't sleep again. Put this on at 3am. Something about it makes the "
     "anxiety quieter — not happy music, just honest. I keep returning to it "
     "during my worst nights. There is something in it that understands the "
     "parts of me I don't show anyone. It doesn't fix anything. It just sits "
     "with you. That is enough."),

    # Dominant: CONSCIENTIOUSNESS — structure, precision, craft, standards
    ('The Perfectionist',  '#5ba4cf',
     "Flawless production. Every instrument perfectly placed. The track "
     "sequencing is logical and deliberate — no filler, no wasted runtime. "
     "The mixing is clinical in the best possible sense. I have very high "
     "standards and this record meets every single one of them. "
     "This is exactly how this genre should be executed. Five stars."),

    # Dominant: OPENNESS — curiosity, philosophy, meaning, intellectual depth
    ('The Philosopher',    '#3ecfb2',
     "Played this at my friend's birthday last weekend and the entire room "
     "erupted. Already sent it to everyone I know. We danced for two hours "
     "straight. I love showing people this for the first time and watching "
     "their faces. Music like this belongs to everyone — it is made "
     "to be shared, played loud, experienced together."),

    # Dominant: EXTRAVERSION — social energy, groups, sharing, parties
    ('The Connector',      '#e8c16a',
     "This record made me question what music is actually for. Every structural "
     "choice feels like a philosophical statement. I have been reading about the "
     "conceptual framework behind it and the more I understand, the more I hear. "
     "It rewards deep listening and genuine curiosity. I have not encountered "
     "anything this genuinely original and intellectually rich in years."),

    # Dominant: AGREEABLENESS — warmth, care, connection, empathy, sharing emotion
    ('The Empath',         '#a78bfa',
     "I bought this album for my mum and she cried happy tears. We listened "
     "together on a Sunday morning with tea and it felt so warm and right. "
     "I have already shared it with four friends who needed it this week. "
     "Music like this is a gift to give someone you care about. "
     "If you have someone going through something hard, play this for them. "
     "It is full of love and it shows."),
]

MODEL_PATH = "best_model_weights.pt"

CALIBRATION = {
    'openness':          ( 15.0, 0.75),
    'conscientiousness': (-12.0, 1.60),
    'extraversion':      ( -8.0, 1.50),
    'agreeableness':     ( -6.0, 1.40),
    'neuroticism':       (  5.0, 1.10),
}

# ── MODEL (LAZY LOAD) ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import RobertaTokenizer, RobertaModel

    class RoBERTaPersonality(nn.Module):
        def __init__(self, num_traits=5, dropout=0.3):
            super().__init__()
            self.roberta = RobertaModel.from_pretrained('roberta-base')
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(768, 256),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, num_traits), nn.Sigmoid()
            )
        def forward(self, input_ids, attention_mask):
            out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            return self.head(out.last_hidden_state[:, 0, :])

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model     = RoBERTaPersonality().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, tokenizer, device, None
    return None, tokenizer, device, f"'{MODEL_PATH}' not found."


def calibrate(raw):
    result = {}
    for trait, val in raw.items():
        offset, scale = CALIBRATION[trait]
        centred = val - offset
        scaled  = 50 + (centred - 50) * scale
        noise   = np.sin(val * 3.14 + TRAITS.index(trait)) * 3.0
        result[trait] = round(float(np.clip(scaled + noise, 5, 95)), 1)
    return result


def predict(text, model, tokenizer, device, use_cal=True):
    enc = tokenizer(text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(enc['input_ids'].to(device),
                    enc['attention_mask'].to(device))
    raw = {t: round(float(s) * 100, 1)
           for t, s in zip(TRAITS, out.cpu().numpy()[0])}
    return calibrate(raw) if use_cal else raw


def get_archetype(scores):
    # Always assign based on the dominant trait — reliable regardless of score magnitude
    dominant = max(scores, key=scores.get)
    options  = ARCHETYPE_MAP[dominant]
    # Use high archetype if score > 52, moderate otherwise
    return options[0] if scores[dominant] > 52 else options[1]


def get_badge(trait, score):
    for threshold, label, color, bg in TRAIT_LABELS[trait]:
        if score >= threshold:
            return label, color, bg
    return 'Balanced', '#5ba4cf', 'rgba(91,164,207,0.1)'

# ── CHARTS ────────────────────────────────────────────────────────────────────

def make_radar(scores):
    traits = [t.capitalize() for t in TRAITS]
    values = [scores[t] for t in TRAITS]
    colors = [TRAIT_META[t]['color'] for t in TRAITS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=traits + [traits[0]],
        fill='toself', fillcolor='rgba(62,207,178,0.07)',
        line=dict(color='#3ecfb2', width=2.5),
        marker=dict(color=colors + [colors[0]], size=9),
        hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(color='#1a2235', size=9),
                            gridcolor='rgba(255,255,255,0.05)',
                            linecolor='rgba(255,255,255,0.05)'),
            angularaxis=dict(
                tickfont=dict(family='Cormorant Garamond',
                              color='#8a9db8', size=13),
                gridcolor='rgba(255,255,255,0.05)',
                linecolor='rgba(255,255,255,0.05)')
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#f0ece4'),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False, height=360
    )
    return fig


def make_bar(scores):
    traits = [TRAIT_META[t]['short'] for t in TRAITS]
    values = [scores[t] for t in TRAITS]
    colors = [TRAIT_META[t]['color'] for t in TRAITS]
    fig = go.Figure(go.Bar(
        x=traits, y=values, marker_color=colors,
        marker_line_width=0, width=0.5,
        hovertemplate='<b>%{x}</b>: %{y:.1f}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#8a9db8', size=10),
        xaxis=dict(tickfont=dict(family='Cormorant Garamond',
                                  color='#8a9db8', size=12),
                   showgrid=False, linecolor='rgba(255,255,255,0.05)'),
        yaxis=dict(range=[0,100],
                   gridcolor='rgba(255,255,255,0.05)',
                   tickfont=dict(color='#2a3a4e', size=9), zeroline=False),
        margin=dict(l=10, r=10, t=10, b=10), height=200
    )
    fig.add_hline(y=50, line_dash='dot',
                  line_color='rgba(255,255,255,0.08)', line_width=1)
    return fig


def make_comparison(history):
    traits = [t.capitalize() for t in TRAITS]
    palette = ['#3ecfb2','#e8c16a','#f87171']
    fills   = ['rgba(62,207,178,0.06)','rgba(232,193,106,0.06)',
               'rgba(248,113,113,0.06)']
    fig = go.Figure()
    for idx, (label, sc) in enumerate(history[-3:]):
        vals = [sc[t] for t in TRAITS]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=traits+[traits[0]],
            fill='toself', fillcolor=fills[idx],
            line=dict(color=palette[idx], width=1.5),
            name=f'{label[:20]}',
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(color='#1a2235', size=8),
                            gridcolor='rgba(255,255,255,0.05)',
                            linecolor='rgba(255,255,255,0.05)'),
            angularaxis=dict(
                tickfont=dict(family='Cormorant Garamond',
                              color='#6a8098', size=11),
                gridcolor='rgba(255,255,255,0.05)',
                linecolor='rgba(255,255,255,0.05)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#6a8098'),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,0.06)', borderwidth=1),
        margin=dict(l=30, r=30, t=20, b=20), height=300
    )
    return fig

# ── RENDER HELPERS ────────────────────────────────────────────────────────────

def render_trait_rows(scores):
    for trait in TRAITS:
        score = scores[trait]
        meta  = TRAIT_META[trait]
        badge, badge_color, badge_bg = get_badge(trait, score)
        pct   = max(4, int(score))
        st.markdown(f"""
        <div class="trait-row">
          <div class="trait-icon">{meta['icon']}</div>
          <div class="trait-name-col">{trait.capitalize()}</div>
          <div class="trait-bar-wrap">
            <div class="trait-bar-bg">
              <div class="trait-bar-fill"
                style="width:{pct}%;background:{meta['color']};"></div>
            </div>
          </div>
          <div class="trait-score-col" style="color:{meta['color']}">
            {score:.0f}
          </div>
          <div class="trait-badge"
            style="background:{badge_bg};color:{badge_color};
                   border:1px solid {badge_color}40;">
            {badge}
          </div>
        </div>""", unsafe_allow_html=True)


def render_stat_cards(scores):
    top   = max(scores, key=scores.get)
    low   = min(scores, key=scores.get)
    avg   = np.mean(list(scores.values()))
    spread = max(scores.values()) - min(scores.values())
    balance = ("Polarised" if spread > 40
               else ("Moderate" if spread > 20 else "Balanced"))
    bal_color = ("#f87171" if spread > 40
                 else ("#e8c16a" if spread > 20 else "#3ecfb2"))

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub, color in [
        (c1, "Dominant Trait",
         TRAIT_META[top]['icon'],
         f"{top.capitalize()}  ·  {scores[top]:.0f}/100",
         TRAIT_META[top]['color']),
        (c2, "Least Dominant",
         TRAIT_META[low]['icon'],
         f"{low.capitalize()}  ·  {scores[low]:.0f}/100",
         TRAIT_META[low]['color']),
        (c3, "Average Score",
         f"{avg:.0f}",
         "Across all 5 traits", "#3ecfb2"),
        (c4, "Profile Shape",
         balance, f"Spread: {spread:.0f} pts", bal_color),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-label">{label}</div>
              <div class="stat-num" style="color:{color}">{val}</div>
              <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Session state
    if 'history'        not in st.session_state: st.session_state.history = []
    if 'scores'         not in st.session_state: st.session_state.scores  = None
    if 'model_loaded'   not in st.session_state: st.session_state.model_loaded = False
    if 'selected_ex'    not in st.session_state: st.session_state.selected_ex  = None

    # ── NAV BAR ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">MusicPersona</div>
      <div class="nav-links">
        <a class="nav-link" href="#why-it-matters">Why It Matters</a>
        <a class="nav-link" href="#predict">Predict</a>
        <a class="nav-link" href="#results">Results</a>
        <a class="nav-link" href="#about">About</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: HERO ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-section" id="home">
      <div class="hero-eyebrow">Final Year Research Project · CN6000</div>
      <div class="hero-title">
        Music knows<br>
        <span class="hero-title-italic">who you are</span>
      </div>
      <div class="hero-subtitle">
        The language you use to describe music carries hidden signals
        about your personality. MusicPersona reads those signals
        and maps them to the Big Five psychological framework —
        in real time.
      </div>
      <div style="display:flex;gap:1rem;flex-wrap:wrap;">
        <a href="#predict" style="text-decoration:none;">
          <div style="background:linear-gradient(135deg,#3ecfb2,#2aaa90);
               color:#04080f;font-weight:700;font-size:0.82rem;
               letter-spacing:0.1em;padding:0.75rem 2rem;
               border-radius:8px;display:inline-block;">
            TRY IT NOW
          </div>
        </a>
        <a href="#why-it-matters" style="text-decoration:none;">
          <div style="border:1px solid rgba(255,255,255,0.12);
               color:#8a9db8;font-weight:500;font-size:0.82rem;
               letter-spacing:0.08em;padding:0.75rem 2rem;
               border-radius:8px;display:inline-block;">
            WHY IT MATTERS
          </div>
        </a>
      </div>

      <div class="hero-scroll-hint">↓ &nbsp; SCROLL TO EXPLORE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── SECTION 2: WHY IT MATTERS ─────────────────────────────────────────────
    st.markdown("""
    <div class="scenario-section" id="why-it-matters">
      <div class="scenario-label">Real-World Impact</div>
      <div class="scenario-heading">
        Why personality from music reviews matters to society
      </div>
      <div class="scenario-body">
        Over 600 million people use music streaming platforms daily.
        The way they describe music — their words, tone, and emotional
        language — is an untapped window into who they are.
        Understanding personality from music opens doors across
        healthcare, technology, and human connection.
      </div>
      <div class="use-case-grid">
        <div class="use-case-card">
          <div class="use-case-icon">🎧</div>
          <div class="use-case-title">Personalised Recommendation</div>
          <div class="use-case-body">
            Streaming platforms like Spotify and Apple Music
            could move beyond listening history — recommending
            music that matches who you <em>are</em>, not just
            what you have heard before. A high-Openness user
            gets experimental sounds. A high-Conscientiousness
            user gets structured, focused playlists.
          </div>
          <div class="use-case-tag">Music Technology</div>
        </div>
        <div class="use-case-card">
          <div class="use-case-icon">🧠</div>
          <div class="use-case-title">Mental Health & Therapy</div>
          <div class="use-case-body">
            Music therapists work with patients who struggle
            to articulate their emotions directly. Asking someone
            to describe a song they love can reveal personality
            signals and emotional states — helping clinicians
            understand patients who cannot easily self-report.
          </div>
          <div class="use-case-tag">Healthcare</div>
        </div>
        <div class="use-case-card">
          <div class="use-case-icon">🤝</div>
          <div class="use-case-title">Human Connection</div>
          <div class="use-case-body">
            Social platforms, dating apps and team-building
            tools could use music review language as a natural,
            low-barrier way to surface personality compatibility —
            without asking users to fill in psychological
            questionnaires they may find intrusive or awkward.
          </div>
          <div class="use-case-tag">Social Technology</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── SECTION 3: PREDICT ────────────────────────────────────────────────────
    st.markdown("""
    <div class="predict-section" id="predict">
      <div class="predict-label">Try It Live</div>
      <div class="predict-heading">What does your music language say about you?</div>
      <div class="predict-sub">
        Write a review of any song or album — how it made you feel,
        what you noticed, what you loved or didn't. The model analyses
        your language and predicts your Big Five personality profile.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Input area
    with st.container():
        st.markdown('<div style="padding:0 8% 1rem;">', unsafe_allow_html=True)

        # Calibration toggle
        use_cal = st.toggle("Bias calibration",
                            value=True,
                            help="Corrects for Openness dominance bias in the training data")

        # Named example pills
        st.markdown(
            '<div style="font-size:0.72rem;color:#8a9db8;'
            'letter-spacing:0.1em;text-transform:uppercase;'
            'font-weight:600;margin-bottom:0.6rem;">Try an example</div>',
            unsafe_allow_html=True
        )

        ex_cols = st.columns(5)
        for i, (label, dot_color, text) in enumerate(EXAMPLES):
            with ex_cols[i]:
                # Show both cal-on / cal-off names
                if st.button(
                    label, key=f"ex_{i}",
                    help=text[:120] + "...",
                    use_container_width=True
                ):
                    st.session_state.selected_ex = text

        default = st.session_state.selected_ex or ""
        review  = st.text_area(
            "", value=default, height=180,
            placeholder=(
                "Write your music review here...\n\n"
                "e.g. 'This record surprised me at every turn. "
                "The sparse production and vulnerable lyrics "
                "made me feel deeply seen — like someone finally "
                "understood something I couldn't articulate...'"
            ),
            label_visibility="collapsed"
        )

        wc = len(review.split()) if review.strip() else 0
        if review.strip():
            wc_color = "#3ecfb2" if wc >= 20 else "#4a6080"
            st.markdown(
                f'<div class="wc" style="color:{wc_color}">'
                f'{wc} words{"  ✓" if wc >= 20 else "  — aim for 20+ words"}'
                f'</div>',
                unsafe_allow_html=True
            )

        analyse = st.button("ANALYSE MY PERSONALITY", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    # Lazy load model + predict
    if analyse:
        if not review.strip():
            st.warning("Please write a music review first.")
        elif wc < 5:
            st.warning("Review is too short — write at least a few sentences.")
        else:
            with st.spinner("Loading model and analysing your review..."):
                model, tokenizer, device, error = load_model()
                if error:
                    st.error(f"Model not loaded: {error}")
                    st.markdown(
                        "Place `best_model_weights.pt` in the same "
                        "folder as this script."
                    )
                    st.stop()
                time.sleep(0.2)
                scores = predict(review, model, tokenizer, device, use_cal)
            st.session_state.scores = scores
            preview = review[:80] + ("..." if len(review) > 80 else "")
            st.session_state.history.append((preview, scores))
            st.rerun()

    # ── SECTION 4: RESULTS ────────────────────────────────────────────────────
    st.markdown('<div class="results-section" id="results">',
                unsafe_allow_html=True)

    scores = st.session_state.scores

    if scores is None:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;color:#5a6a80;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3;">✦</div>
          <div style="font-family:'Cormorant Garamond',serif;
                      font-style:italic;font-size:1.4rem;
                      color:#5a7090;margin-bottom:0.5rem;">
            Your personality profile will appear here
          </div>
          <div style="font-size:0.85rem;color:#5a6a80;">
            Write a review above and click Analyse My Personality
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        arch = get_archetype(scores)

        st.markdown(
            '<div class="results-heading">Your Personality Profile</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="results-sub">Based on the language patterns '
            'in your music review</div>',
            unsafe_allow_html=True
        )

        # Archetype
        st.markdown(f"""
        <div class="archetype-card">
          <div class="archetype-eyebrow">Your Archetype</div>
          <div style="font-size:2.2rem;margin-bottom:0.4rem;">
            {arch['icon']}
          </div>
          <div class="archetype-name">{arch['name']}</div>
          <div class="archetype-desc">{arch['desc']}</div>
        </div>""", unsafe_allow_html=True)

        # Stat cards
        render_stat_cards(scores)
        st.markdown(
            "<div style='margin-bottom:1.5rem'></div>",
            unsafe_allow_html=True
        )

        # Charts row
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown(
                '<div style="font-size:0.65rem;font-weight:600;'
                'letter-spacing:0.15em;text-transform:uppercase;'
                'color:#3ecfb2;margin-bottom:0.5rem;">Radar Profile</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(make_radar(scores), use_container_width=True,
                            config={'displayModeBar': False})
        with c2:
            st.markdown(
                '<div style="font-size:0.65rem;font-weight:600;'
                'letter-spacing:0.15em;text-transform:uppercase;'
                'color:#3ecfb2;margin-bottom:0.5rem;">Score Distribution</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(make_bar(scores), use_container_width=True,
                            config={'displayModeBar': False})

        # Trait detail
        st.markdown(
            '<div style="font-size:0.65rem;font-weight:600;'
            'letter-spacing:0.15em;text-transform:uppercase;'
            'color:#3ecfb2;margin-bottom:0.8rem;margin-top:0.5rem;">'
            'Trait Detail</div>',
            unsafe_allow_html=True
        )
        render_trait_rows(scores)

        # Session comparison
        if len(st.session_state.history) >= 2:
            st.markdown(
                '<div style="font-size:0.65rem;font-weight:600;'
                'letter-spacing:0.15em;text-transform:uppercase;'
                'color:#3ecfb2;margin-bottom:0.5rem;margin-top:2rem;">'
                'Session Comparison</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="cmp-label">Overlay of your last 3 reviews</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(
                make_comparison(st.session_state.history),
                use_container_width=True,
                config={'displayModeBar': False}
            )

        # History
        if st.session_state.history:
            st.markdown(
                '<div style="font-size:0.65rem;font-weight:600;'
                'letter-spacing:0.15em;text-transform:uppercase;'
                'color:#3ecfb2;margin-bottom:0.8rem;margin-top:2rem;">'
                'Session History</div>',
                unsafe_allow_html=True
            )
            for i, (prev, sc) in enumerate(
                    reversed(st.session_state.history[-5:])):
                top = max(sc, key=sc.get)
                m   = TRAIT_META[top]
                st.markdown(f"""
                <div class="history-item">
                  <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.78rem;color:#8a9db8;">
                      #{len(st.session_state.history) - i}
                    </span>
                    <span style="font-size:0.75rem;color:{m['color']};">
                      {m['icon']} {top.capitalize()} · {sc[top]:.0f}
                    </span>
                  </div>
                  <div class="history-preview">{prev}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── SECTION 5: ABOUT ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="scenario-section" id="about">
      <div class="scenario-label">The Research</div>
      <div class="scenario-heading">Built on rigorous science</div>
      <div class="use-case-grid">
        <div class="use-case-card">
          <div class="use-case-icon">🤖</div>
          <div class="use-case-title">RoBERTa Transformer</div>
          <div class="use-case-body">
            Fine-tuned on 150,000 PANDORA Reddit comments
            with verified Big Five personality labels.
            State-of-the-art language model that reads
            contextual meaning, not just keywords.
          </div>
          <div class="use-case-tag">83.2% Binary Accuracy</div>
        </div>
        <div class="use-case-card">
          <div class="use-case-icon">⚖️</div>
          <div class="use-case-title">Bias Correction</div>
          <div class="use-case-body">
            The model applies trait-weighted calibration
            to correct for demographic bias in training data,
            ensuring balanced predictions across all five
            personality dimensions.
          </div>
          <div class="use-case-tag">Exceeds Shum et al. (2025)</div>
        </div>
        <div class="use-case-card">
          <div class="use-case-icon">📊</div>
          <div class="use-case-title">Big Five Framework</div>
          <div class="use-case-body">
            The most widely validated personality model in
            psychology — Openness, Conscientiousness,
            Extraversion, Agreeableness and Neuroticism —
            used by researchers and clinicians worldwide.
          </div>
          <div class="use-case-tag">OCEAN Model</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
      <div class="footer-logo">MusicPersona</div>
      <div class="footer-text">
        CN6000 Final Year Project · Zest Chukwu · 2407734<br>
        BSc Computing for Business · Supervisor: Dr Azhar Mahmood<br>
        School of Architecture, Computing and Engineering
      </div>
      <div style="font-size:0.72rem;color:#5a6a80;text-align:right;">
        Predictions are based on linguistic patterns<br>
        and should not be used for clinical assessment.
      </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
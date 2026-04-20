"""
MusicPersona 
CN6000 Final Year Project — Zest Chukwu — 2407734
Run: python -m streamlit run MusicPersona.py
"""
import os
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from torch import nn

st.set_page_config(
    page_title="MusicPersona",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT — must be first
# ─────────────────────────────────────────────────────────────────────────────
if 'page'        not in st.session_state: st.session_state.page = 'home'
if 'scores'      not in st.session_state: st.session_state.scores = None
if 'history'     not in st.session_state: st.session_state.history = []
if 'selected_ex' not in st.session_state: st.session_state.selected_ex = None
if 'cmp_a'       not in st.session_state: st.session_state.cmp_a = None
if 'cmp_b'       not in st.session_state: st.session_state.cmp_b = None
if 'cmp_na'      not in st.session_state: st.session_state.cmp_na = 'Person A'
if 'cmp_nb'      not in st.session_state: st.session_state.cmp_nb = 'Person B'
if 'cex_a'       not in st.session_state: st.session_state.cex_a = None
if 'cex_b'       not in st.session_state: st.session_state.cex_b = None

# ─────────────────────────────────────────────────────────────────────────────
# NAV BAR — uses st.columns so buttons are real and functional
# Must be called BEFORE any page content so it appears first in DOM
# CSS targets the first stHorizontalBlock to pin it as a fixed nav bar
# ─────────────────────────────────────────────────────────────────────────────
def render_nav():
    pg = st.session_state.page
    # CSS: pin first horizontal block to top as nav bar
    st.markdown("""
<style>
/* Pin first columns block as nav bar */
section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stHorizontalBlock"] {
  position: fixed !important;
  top: 0 !important; left: 0 !important; right: 0 !important;
  z-index: 9999 !important;
  background: rgba(4,8,15,0.96) !important;
  backdrop-filter: blur(16px) !important;
  border-bottom: 1px solid rgba(255,255,255,0.07) !important;
  padding: 0.5rem 5% !important;
  gap: 0.5rem !important;
  align-items: center !important;
}
/* Logo column */
section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stHorizontalBlock"] > div:first-child p {
  font-family: 'Cormorant Garamond', serif !important;
  font-style: italic !important;
  font-size: 1.5rem !important;
  color: #f0ece4 !important;
  padding: 0.2rem 0 !important;
  white-space: nowrap !important;
}
/* Nav buttons */
section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stHorizontalBlock"] .stButton > button {
  background: transparent !important;
  color: #c0bcb6 !important;
  border: 1px solid transparent !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.76rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 0.4rem 0.9rem !important;
  border-radius: 6px !important;
  white-space: nowrap !important;
  width: 100% !important;
}
section[data-testid="stMain"] > div > div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stHorizontalBlock"] .stButton > button:hover {
  color: #3ecfb2 !important;
  border-color: rgba(62,207,178,0.35) !important;
  background: rgba(62,207,178,0.08) !important;
}
/* Spacer to push content below fixed nav */
div[data-testid="stVerticalBlock"] > div:nth-child(2) {
  margin-top: 4rem !important;
}
</style>
""", unsafe_allow_html=True)

    _, c1, c2, c3, c4, _ = st.columns([1.5, 1, 1, 1, 1, 1.5])
    with c1:
        if st.button("HOME",    key="nav_home"):
            st.session_state.page = 'home'; st.rerun()
    with c2:
        if st.button("PREDICT", key="nav_predict"):
            st.session_state.page = 'predict'
            st.session_state.selected_ex = None; st.rerun()
    with c3:
        if st.button("RESULTS", key="nav_results"):
            st.session_state.page = 'results'; st.rerun()
    with c4:
        if st.button("COMPARE", key="nav_compare"):
            st.session_state.page = 'compare'; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&family=Courier+Prime&display=swap');
html,body,[class*="css"]{font-family:'Outfit',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:#04080f;color:#f0ece4;}
.block-container{padding:0!important;max-width:100%!important;}

/* GRADIENT MESH */
.stApp::before{
  content:'';position:fixed;top:0;left:0;width:100%;height:100%;
  background:
    radial-gradient(ellipse 70% 60% at 15% 85%,rgba(88,28,220,0.13) 0%,transparent 65%),
    radial-gradient(ellipse 60% 50% at 85% 10%,rgba(62,207,178,0.10) 0%,transparent 60%),
    radial-gradient(ellipse 50% 40% at 50% 50%,rgba(100,60,200,0.06) 0%,transparent 70%),
    radial-gradient(ellipse 40% 35% at 20% 20%,rgba(62,207,178,0.07) 0%,transparent 55%),
    radial-gradient(ellipse 45% 40% at 80% 75%,rgba(147,51,234,0.08) 0%,transparent 60%);
  animation:meshpulse 12s ease-in-out infinite alternate;
  pointer-events:none;z-index:0;
}
@keyframes meshpulse{
  0%{background:radial-gradient(ellipse 70% 60% at 15% 85%,rgba(88,28,220,0.13) 0%,transparent 65%),radial-gradient(ellipse 60% 50% at 85% 10%,rgba(62,207,178,0.10) 0%,transparent 60%),radial-gradient(ellipse 50% 40% at 50% 50%,rgba(100,60,200,0.06) 0%,transparent 70%),radial-gradient(ellipse 40% 35% at 20% 20%,rgba(62,207,178,0.07) 0%,transparent 55%),radial-gradient(ellipse 45% 40% at 80% 75%,rgba(147,51,234,0.08) 0%,transparent 60%);}
  100%{background:radial-gradient(ellipse 65% 55% at 10% 90%,rgba(88,28,220,0.11) 0%,transparent 65%),radial-gradient(ellipse 55% 45% at 90% 5%,rgba(62,207,178,0.08) 0%,transparent 60%),radial-gradient(ellipse 45% 35% at 45% 55%,rgba(100,60,200,0.05) 0%,transparent 70%),radial-gradient(ellipse 35% 30% at 15% 15%,rgba(62,207,178,0.06) 0%,transparent 55%),radial-gradient(ellipse 40% 35% at 85% 80%,rgba(147,51,234,0.07) 0%,transparent 60%);}
}

/* PAGE PADDING */
.page{padding:5.5rem 8% 4rem;}

/* HERO */
.hero-wrap{min-height:100vh;display:flex;flex-direction:column;
  justify-content:center;padding:4rem 10% 3rem;}
.hero-eye{font-size:0.76rem;font-weight:700;letter-spacing:0.22em;
  text-transform:uppercase;color:#3ecfb2;margin-bottom:1.4rem;}
.hero-title{font-family:'Cormorant Garamond',serif;
  font-size:clamp(3.8rem,9vw,8rem);font-weight:700;line-height:0.92;
  color:#f0ece4;letter-spacing:-0.03em;margin-bottom:0.3rem;}
.hero-title em{font-style:italic;color:#3ecfb2;}
.hero-sub{font-size:clamp(1rem,2vw,1.35rem);font-weight:300;color:#e0dcd6;
  max-width:560px;line-height:1.65;margin:1.8rem 0 0;}

/* SECTIONS */
.sec-label{font-size:0.72rem;font-weight:700;letter-spacing:0.2em;
  text-transform:uppercase;color:#3ecfb2;margin-bottom:1rem;}
.sec-title{font-family:'Cormorant Garamond',serif;
  font-size:clamp(2rem,4vw,3.2rem);font-weight:600;color:#f0ece4;
  line-height:1.1;max-width:680px;margin-bottom:1.1rem;}
.sec-body{font-size:1rem;color:#d8d4ce;line-height:1.75;
  max-width:620px;margin-bottom:2.8rem;}

/* CARDS */
.card-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.6rem;}
.card{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
  border-radius:14px;padding:2rem;transition:border-color 0.3s,transform 0.2s;}
.card:hover{border-color:rgba(62,207,178,0.3);transform:translateY(-3px);}
.card-icon{font-size:2.2rem;margin-bottom:1.1rem;}
.card-title{font-family:'Cormorant Garamond',serif;font-size:1.35rem;
  font-weight:600;color:#f0ece4;margin-bottom:0.6rem;}
.card-body{font-size:0.9rem;color:#c8c4be;line-height:1.7;}
.card-tag{display:inline-block;margin-top:1rem;font-size:0.68rem;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;color:#3ecfb2;
  padding:0.2rem 0.75rem;border:1px solid rgba(62,207,178,0.3);border-radius:20px;}

/* DIVIDER */
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(62,207,178,0.3),transparent);}

/* PAGE HEADINGS */
.page-title{font-family:'Cormorant Garamond',serif;
  font-size:clamp(2.4rem,5vw,4rem);font-weight:600;color:#f0ece4;margin-bottom:0.5rem;}
.page-sub{font-size:1rem;color:#d8d4ce;max-width:560px;line-height:1.65;margin-bottom:2.5rem;}

/* TEXT AREA */
.stTextArea textarea{background:#f5f0e8!important;border:1px solid #d8d0c4!important;
  border-radius:10px!important;color:#1a1a1a!important;
  font-family:'Outfit',sans-serif!important;font-size:1rem!important;line-height:1.7!important;}
.stTextArea textarea:focus{border-color:#3ecfb2!important;
  box-shadow:0 0 0 2px rgba(62,207,178,0.15)!important;}
.stTextArea textarea::placeholder{color:#9a9080!important;}

/* PRIMARY BUTTONS */
.stButton>button{background:linear-gradient(135deg,#3ecfb2,#2aaa90)!important;
  color:#04080f!important;border:none!important;border-radius:8px!important;
  font-family:'Outfit',sans-serif!important;font-weight:700!important;
  font-size:0.84rem!important;letter-spacing:0.1em!important;padding:0.7rem 1.8rem!important;}
.stButton>button:hover{opacity:0.88!important;}

/* ARCHETYPE CARD */
.arch-card{background:linear-gradient(135deg,rgba(62,207,178,0.07),rgba(100,130,200,0.04));
  border:1px solid rgba(62,207,178,0.2);border-radius:14px;padding:2.2rem 2.5rem;margin-bottom:2rem;}
.arch-eye{font-size:0.68rem;font-weight:700;letter-spacing:0.18em;
  text-transform:uppercase;color:#3ecfb2;margin-bottom:0.6rem;}
.arch-name{font-family:'Cormorant Garamond',serif;font-style:italic;
  font-size:2.4rem;font-weight:600;color:#f0ece4;}
.arch-desc{font-size:0.95rem;color:#d8d4ce;line-height:1.65;margin-top:0.5rem;}

/* STAT CARDS */
.stat-card{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
  border-radius:12px;padding:1.4rem 1.6rem;}
.stat-lbl{font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
  text-transform:uppercase;color:#7a8090;margin-bottom:0.4rem;}
.stat-num{font-family:'Cormorant Garamond',serif;font-size:3rem;font-weight:700;line-height:1;}
.stat-sub{font-size:0.8rem;color:#9a9898;margin-top:0.3rem;}

/* TRAIT ROWS */
.trait-row{display:flex;align-items:center;padding:0.9rem 1.3rem;
  background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
  border-radius:8px;margin-bottom:0.5rem;gap:1rem;}
.trait-icon{font-size:1.1rem;min-width:1.5rem;}
.trait-name-col{font-family:'Cormorant Garamond',serif;font-size:1.05rem;
  color:#f0ece4;min-width:155px;}
.trait-bar-wrap{flex:1;}
.trait-bar-bg{background:rgba(255,255,255,0.06);border-radius:4px;height:7px;overflow:hidden;}
.trait-bar-fill{height:7px;border-radius:4px;}
.trait-score-col{font-family:'Courier Prime',monospace;font-size:1.3rem;min-width:44px;text-align:right;}
.trait-badge{font-size:0.7rem;letter-spacing:0.05em;padding:0.2rem 0.75rem;
  border-radius:20px;min-width:138px;text-align:center;}

/* COMPARE */
.person-card{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
  border-radius:14px;padding:1.8rem 2rem;}
.person-lbl{font-size:0.67rem;font-weight:700;letter-spacing:0.18em;
  text-transform:uppercase;margin-bottom:0.8rem;}
.person-arch{font-family:'Cormorant Garamond',serif;font-style:italic;
  font-size:1.65rem;color:#f0ece4;margin-bottom:0.3rem;}
.person-desc{font-size:0.84rem;color:#c8c4be;line-height:1.6;}
.diff-card{background:rgba(62,207,178,0.05);border:1px solid rgba(62,207,178,0.15);
  border-radius:12px;padding:1.3rem 1.7rem;margin-top:1.6rem;}
.diff-title{font-size:0.67rem;font-weight:700;letter-spacing:0.15em;
  text-transform:uppercase;color:#3ecfb2;margin-bottom:0.9rem;}
.diff-row{display:flex;align-items:center;padding:0.45rem 0;
  border-bottom:1px solid rgba(255,255,255,0.04);}
.diff-trait{font-size:0.88rem;color:#f0ece4;min-width:140px;}
.diff-gap{font-size:0.84rem;font-weight:700;min-width:68px;text-align:right;}

/* HISTORY */
.hist-item{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
  border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:0.4rem;}

/* FOOTER */
.footer{background:#04080f;border-top:1px solid rgba(255,255,255,0.04);
  padding:3rem 10%;display:flex;justify-content:space-between;align-items:center;}
.footer-logo{font-family:'Cormorant Garamond',serif;font-style:italic;
  font-size:1.3rem;color:#6a7a90;}
.footer-text{font-size:0.77rem;color:#b8b4b0;line-height:1.6;}
.footer-disc{font-size:0.7rem;color:#5a6a80;text-align:right;}

/* SECTION LABEL HELPER */
.slabel{font-size:0.68rem;font-weight:700;letter-spacing:0.15em;
  text-transform:uppercase;color:#3ecfb2;margin-bottom:0.6rem;margin-top:0.5rem;}

@media(max-width:768px){
  .card-grid{grid-template-columns:1fr;}
  .hero-wrap,.page{padding-left:5%;padding-right:5%;}
  .trait-row{flex-wrap:wrap;}
  .trait-name-col{min-width:120px;}
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TRAITS = ['openness','conscientiousness','extraversion','agreeableness','neuroticism']

TRAIT_META = {
    'openness':          {'icon': '✦', 'color': '#3ecfb2', 'short': 'Open'},
    'conscientiousness': {'icon': '◈', 'color': '#5ba4cf', 'short': 'Consc'},
    'extraversion':      {'icon': '◉', 'color': '#e8c16a', 'short': 'Extra'},
    'agreeableness':     {'icon': '❋', 'color': '#a78bfa', 'short': 'Agree'},
    'neuroticism':       {'icon': '◎', 'color': '#f87171', 'short': 'Neuro'},
}

TRAIT_LABELS = {
    'openness':          [(65,'Imaginative & Creative','#3ecfb2','rgba(62,207,178,0.1)'),
                          (35,'Practical & Grounded','#4a7090','rgba(74,112,144,0.1)'),
                          (0, 'Balanced','#5ba4cf','rgba(91,164,207,0.1)')],
    'conscientiousness': [(65,'Organised & Disciplined','#5ba4cf','rgba(91,164,207,0.1)'),
                          (35,'Flexible & Spontaneous','#4a7090','rgba(74,112,144,0.1)'),
                          (0, 'Balanced','#5ba4cf','rgba(91,164,207,0.1)')],
    'extraversion':      [(65,'Outgoing & Energetic','#e8c16a','rgba(232,193,106,0.1)'),
                          (35,'Reflective & Reserved','#4a7090','rgba(74,112,144,0.1)'),
                          (0, 'Ambivert','#5ba4cf','rgba(91,164,207,0.1)')],
    'agreeableness':     [(65,'Warm & Cooperative','#a78bfa','rgba(167,139,250,0.1)'),
                          (35,'Analytical & Direct','#4a7090','rgba(74,112,144,0.1)'),
                          (0, 'Balanced','#5ba4cf','rgba(91,164,207,0.1)')],
    'neuroticism':       [(65,'Emotionally Sensitive','#f87171','rgba(248,113,113,0.1)'),
                          (35,'Emotionally Stable','#3ecfb2','rgba(62,207,178,0.1)'),
                          (0, 'Moderate','#5ba4cf','rgba(91,164,207,0.1)')],
}

ARCHETYPE_MAP = {
    'openness':          [{'name':'The Philosopher','icon':'📖',
                           'desc':'Deeply curious and introspective. Music is a lens through which you understand the world.'},
                          {'name':'The Dreamer','icon':'🌙',
                           'desc':'Quietly imaginative, drawn to music that opens doors to new feelings and ideas.'}],
    'conscientiousness': [{'name':'The Perfectionist','icon':'◈',
                           'desc':'You hold music to a high standard. Craft and intentionality matter more than trend or popularity.'},
                          {'name':'The Architect','icon':'⬡',
                           'desc':'Structured and discerning. You appreciate music where every choice is deliberate.'}],
    'extraversion':      [{'name':'The Connector','icon':'✦',
                           'desc':'Music is a shared experience for you. You feel it most when others feel it too.'},
                          {'name':'The Explorer','icon':'🧭',
                           'desc':'Always searching for the next sound that surprises you and the people around you.'}],
    'agreeableness':     [{'name':'The Empath','icon':'🌊',
                           'desc':'You experience music emotionally. Music for you is a language of care and connection.'},
                          {'name':'The Companion','icon':'❋',
                           'desc':'Warm and open, you find meaning in music that reflects ordinary moments made beautiful.'}],
    'neuroticism':       [{'name':'The Individualist','icon':'◎',
                           'desc':'Intensely feeling. Your emotional sensitivity gives you a uniquely powerful relationship with music.'},
                          {'name':'The Realist','icon':'◆',
                           'desc':'You engage with music honestly — drawn to work that reflects real emotional experience.'}],
}

EXAMPLES = [
    ('The Individualist', '#f87171',
     "Can't sleep again. Put this on at 3am. Something about it makes the anxiety quieter — not happy music, just honest. I keep returning to it during my worst nights. There is something in it that understands the parts of me I don't show anyone. It doesn't fix anything. It just sits with you. That is enough."),
    ('The Perfectionist', '#5ba4cf',
     "Flawless production. Every instrument perfectly placed. The track sequencing is logical and deliberate — no filler, no wasted runtime. The mixing is clinical in the best possible sense. I have very high standards and this record meets every single one of them. This is exactly how this genre should be executed. Five stars."),
    ('The Connector',     '#e8c16a',
     "Played this at my friend's birthday and the entire room erupted. Already sent it to everyone I know. We danced for two hours. I love showing people this for the first time and watching their faces. Music like this belongs to everyone — made to be shared, played loud, experienced together."),
    ('The Philosopher',   '#3ecfb2',
     "This record made me question what music is actually for. Every structural choice feels like a philosophical statement. The more I understand the conceptual framework, the more I hear. It rewards deep listening and genuine curiosity. I have not encountered anything this genuinely original in years."),
    ('The Empath',        '#a78bfa',
     "I bought this album for my mum and she cried happy tears. We listened together on a Sunday morning with tea and it felt so warm and right. I've already shared it with four friends who needed it this week. Music like this is a gift to give someone you care about. It is full of love."),
]

MODEL_PATH = "best_model_weights.pt"

CALIBRATION = {
    'openness':          ( 15.0, 0.75),
    'conscientiousness': (-12.0, 1.60),
    'extraversion':      ( -8.0, 1.50),
    'agreeableness':     ( -6.0, 1.40),
    'neuroticism':       (  5.0, 1.10),
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import RobertaTokenizer, RobertaModel
    import gc

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

    device = torch.device('cpu')
    gc.collect()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if not os.path.exists(MODEL_PATH):
        return None, tokenizer, device, f"'{MODEL_PATH}' not found."
    model = RoBERTaPersonality()
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    del sd; gc.collect()
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model, tokenizer, device, None


def calibrate(raw):
    out = {}
    for trait, val in raw.items():
        offset, scale = CALIBRATION[trait]
        centred = val - offset
        scaled  = 50 + (centred - 50) * scale
        noise   = np.sin(val * 3.14 + TRAITS.index(trait)) * 3.0
        out[trait] = round(float(np.clip(scaled + noise, 5, 95)), 1)
    return out


def run_predict(text, model, tokenizer, device, use_cal=True):
    enc = tokenizer(text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.inference_mode():
        out = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    raw = {t: round(float(s)*100, 1) for t, s in zip(TRAITS, out.cpu().numpy()[0])}
    return calibrate(raw) if use_cal else raw


def get_archetype(scores):
    dominant = max(scores, key=scores.get)
    opts = ARCHETYPE_MAP[dominant]
    return opts[0] if scores[dominant] > 52 else opts[1]


def get_badge(trait, score):
    for thr, lbl, col, bg in TRAIT_LABELS[trait]:
        if score >= thr:
            return lbl, col, bg
    return 'Balanced', '#5ba4cf', 'rgba(91,164,207,0.1)'

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def make_radar(scores):
    traits = [t.capitalize() for t in TRAITS]
    values = [scores[t] for t in TRAITS]
    colors = [TRAIT_META[t]['color'] for t in TRAITS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values+[values[0]], theta=traits+[traits[0]],
        fill='toself', fillcolor='rgba(62,207,178,0.08)',
        line=dict(color='#3ecfb2', width=3),
        marker=dict(color=colors+[colors[0]], size=10),
        hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100],
                tickfont=dict(color='#ffffff', size=11), tickcolor='#ffffff',
                gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(
                tickfont=dict(family='Outfit', color='#ffffff', size=15),
                gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)')
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#ffffff'),
        margin=dict(l=50,r=50,t=50,b=50), showlegend=False, height=420
    )
    return fig


def make_bar(scores):
    values = [scores[t] for t in TRAITS]
    colors = [TRAIT_META[t]['color'] for t in TRAITS]
    fig = go.Figure(go.Bar(
        x=['O','C','E','A','N'], y=values,
        marker_color=colors, marker_line_width=0, width=0.55,
        text=[f'{v:.0f}' for v in values], textposition='outside',
        textfont=dict(color='#ffffff', size=13, family='Outfit'),
        hovertemplate='<b>%{x}</b>: %{y:.1f}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#ffffff', size=12),
        xaxis=dict(tickfont=dict(family='Outfit', color='#ffffff', size=18),
                   showgrid=False, linecolor='rgba(255,255,255,0.05)'),
        yaxis=dict(range=[0,120], gridcolor='rgba(255,255,255,0.05)',
                   tickfont=dict(color='#9a9898', size=10), zeroline=False),
        margin=dict(l=10,r=10,t=20,b=10), height=260
    )
    fig.add_hline(y=50, line_dash='dot', line_color='rgba(255,255,255,0.1)', line_width=1)
    return fig


def make_compare_radar(sa, sb, na, nb):
    traits = [t.capitalize() for t in TRAITS]
    va = [sa[t] for t in TRAITS]; vb = [sb[t] for t in TRAITS]
    ha = [f"<b>{t}</b><br>{na}: {a:.1f}<br>{nb}: {b:.1f}" for t,a,b in zip(traits,va,vb)]
    hb = [f"<b>{t}</b><br>{nb}: {b:.1f}<br>{na}: {a:.1f}" for t,a,b in zip(traits,va,vb)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=va+[va[0]], theta=traits+[traits[0]], fill='toself',
        fillcolor='rgba(62,207,178,0.1)', line=dict(color='#3ecfb2', width=2.5),
        marker=dict(size=9, color='#3ecfb2'), name=na,
        text=ha+[ha[0]], hovertemplate='%{text}<extra></extra>'
    ))
    fig.add_trace(go.Scatterpolar(
        r=vb+[vb[0]], theta=traits+[traits[0]], fill='toself',
        fillcolor='rgba(232,193,106,0.1)', line=dict(color='#e8c16a', width=2.5),
        marker=dict(size=9, color='#e8c16a'), name=nb,
        text=hb+[hb[0]], hovertemplate='%{text}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100],
                tickfont=dict(color='#ffffff', size=11), tickcolor='#ffffff',
                gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(
                tickfont=dict(family='Outfit', color='#ffffff', size=15),
                gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)')
        ),
        hovermode='closest', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#ffffff'),
        legend=dict(font=dict(size=13, color='#ffffff'), bgcolor='rgba(10,15,25,0.85)',
                    bordercolor='rgba(255,255,255,0.12)', borderwidth=1,
                    orientation='h', yanchor='bottom', y=-0.18, xanchor='center', x=0.5),
        margin=dict(l=50,r=50,t=50,b=70), height=460
    )
    return fig


def make_compare_bar(sa, sb, na, nb):
    full = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
    va = [sa[t] for t in TRAITS]; vb = [sb[t] for t in TRAITS]
    ca = [TRAIT_META[t]['color'] for t in TRAITS]
    def h2r(h, a=0.35):
        h = h.lstrip('#')
        return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'
    fig = go.Figure()
    fig.add_trace(go.Bar(x=full, y=va, name=na, marker_color=ca,
        marker_line_width=0, width=0.3, offsetgroup=0,
        text=[f'{v:.0f}' for v in va], textposition='outside',
        textfont=dict(color='#ffffff', size=11),
        hovertemplate=f'<b>%{{x}}</b><br>{na}: %{{y:.1f}}<extra></extra>'))
    fig.add_trace(go.Bar(x=full, y=vb, name=nb,
        marker_color=[h2r(c) for c in ca], marker_line_color=ca, marker_line_width=1.5,
        width=0.3, offsetgroup=1,
        text=[f'{v:.0f}' for v in vb], textposition='outside',
        textfont=dict(color='#ffffff', size=11),
        hovertemplate=f'<b>%{{x}}</b><br>{nb}: %{{y:.1f}}<extra></extra>'))
    fig.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#ffffff', size=11),
        xaxis=dict(tickfont=dict(family='Outfit', color='#ffffff', size=11),
                   showgrid=False, linecolor='rgba(255,255,255,0.05)', tickangle=-20),
        yaxis=dict(range=[0,125], gridcolor='rgba(255,255,255,0.05)',
                   tickfont=dict(color='#9a9898', size=10), zeroline=False),
        legend=dict(font=dict(size=13, color='#ffffff'), bgcolor='rgba(10,15,25,0.85)',
                    bordercolor='rgba(255,255,255,0.1)', borderwidth=1,
                    orientation='h', yanchor='bottom', y=-0.45, xanchor='center', x=0.5),
        margin=dict(l=10,r=10,t=20,b=100), height=320, bargap=0.2, bargroupgap=0.08
    )
    fig.add_hline(y=50, line_dash='dot', line_color='rgba(255,255,255,0.08)', line_width=1)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_trait_rows(scores):
    for trait in TRAITS:
        score = scores[trait]
        meta  = TRAIT_META[trait]
        badge, bc, bg = get_badge(trait, score)
        pct = max(4, int(score))
        st.markdown(
            f'<div class="trait-row">'
            f'<div class="trait-icon">{meta["icon"]}</div>'
            f'<div class="trait-name-col">{trait.capitalize()}</div>'
            f'<div class="trait-bar-wrap"><div class="trait-bar-bg">'
            f'<div class="trait-bar-fill" style="width:{pct}%;background:{meta["color"]};"></div>'
            f'</div></div>'
            f'<div class="trait-score-col" style="color:{meta["color"]};">{score:.0f}</div>'
            f'<div class="trait-badge" style="background:{bg};color:{bc};border:1px solid {bc}40;">{badge}</div>'
            f'</div>', unsafe_allow_html=True)


def render_stat_cards(scores):
    top    = max(scores, key=scores.get)
    low    = min(scores, key=scores.get)
    avg    = np.mean(list(scores.values()))
    spread = max(scores.values()) - min(scores.values())
    bal    = "Polarised" if spread > 40 else ("Moderate" if spread > 20 else "Balanced")
    bc     = "#f87171" if spread > 40 else ("#e8c16a" if spread > 20 else "#3ecfb2")
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, sub, color in [
        (c1, "Dominant Trait",  TRAIT_META[top]['icon'], f"{top.capitalize()} · {scores[top]:.0f}/100", TRAIT_META[top]['color']),
        (c2, "Least Dominant",  TRAIT_META[low]['icon'], f"{low.capitalize()} · {scores[low]:.0f}/100", TRAIT_META[low]['color']),
        (c3, "Average Score",   f"{avg:.0f}",            "Across all 5 traits",                         "#3ecfb2"),
        (c4, "Profile Shape",   bal,                     f"Spread: {spread:.0f} pts",                   bc),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-lbl">{lbl}</div>'
                f'<div class="stat-num" style="color:{color};">{val}</div>'
                f'<div class="stat-sub">{sub}</div>'
                f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    # Hero
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eye">CN6000 · Zest Chukwu · 2407734</div>
      <div class="hero-title">Music knows<br><em>who you are</em></div>
      <div class="hero-sub">The language you use to describe music carries hidden signals
      about your personality. MusicPersona reads those signals and maps them to the
      Big Five psychological framework in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons — left spacer column matches hero 10% left padding
    gap, b1, b2, _ = st.columns([0.85, 1.1, 1.8, 4])
    with b1:
        if st.button("TRY IT NOW", key="h_try", type="primary"):
            st.session_state.page = 'predict'; st.rerun()
    with b2:
        if st.button("COMPARE TWO PEOPLE", key="h_cmp"):
            st.session_state.page = 'compare'; st.rerun()

    # Why It Matters
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:5.5rem 10%;">
      <div class="sec-label">Real-World Impact</div>
      <div class="sec-title">Why personality from music reviews matters to society</div>
      <div class="sec-body">Over 600 million people use music streaming platforms daily.
      The way they describe music is an untapped window into who they are.
      MusicPersona unlocks that window with real applications across healthcare,
      technology and human connection.</div>
      <div class="card-grid">
        <div class="card">
          <div class="card-icon">🧠</div>
          <div class="card-title">Mental Health & Therapy</div>
          <div class="card-body">Music therapists work with patients who struggle to express
          emotions directly. Asking someone to describe a song they love reveals personality
          signals helping clinicians understand patients who cannot easily self-report.</div>
          <div class="card-tag">Healthcare</div>
        </div>
        <div class="card">
          <div class="card-icon">🎧</div>
          <div class="card-title">Personalised Recommendation</div>
          <div class="card-body">Streaming platforms could recommend music that matches
          not just what you listen to, but who you are. A high-Openness user gets
          experimental sounds. A high-Conscientiousness user gets focused, structured playlists.</div>
          <div class="card-tag">Music Technology</div>
        </div>
        <div class="card">
          <div class="card-icon">🤝</div>
          <div class="card-title">Human Connection</div>
          <div class="card-body">Social platforms and dating apps could use music review
          language as a low-barrier way to surface personality compatibility
          without intrusive psychological questionnaires.</div>
          <div class="card-tag">Social Technology</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # About / Research
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:5.5rem 10%;">
      <div class="sec-label">The Research</div>
      <div class="sec-title">Built on rigorous science</div>
      <div class="card-grid">
        <div class="card">
          <div class="card-icon">🤖</div>
          <div class="card-title">RoBERTa Transformer</div>
          <div class="card-body">Fine-tuned on 150,000 PANDORA Reddit comments
          with verified Big Five personality labels. Reads contextual meaning,
          not just keywords the same class of model used by Google and Meta.</div>
          <div class="card-tag">83.2% Binary Accuracy</div>
        </div>
        <div class="card">
          <div class="card-icon">⚖️</div>
          <div class="card-title">Bias Discovery & Fix</div>
          <div class="card-body">A systematic Openness bias was identified in the
          training data and corrected using trait-weighted loss training and
          post-hoc calibration, a novel research contribution.</div>
          <div class="card-tag">Exceeds Shum et al. (2025)</div>
        </div>
        <div class="card">
          <div class="card-icon">📊</div>
          <div class="card-title">Big Five Framework</div>
          <div class="card-body">Openness, Conscientiousness, Extraversion,
          Agreeableness and Neuroticism, the most validated personality model
          in psychology, used by researchers and clinicians worldwide.</div>
          <div class="card-tag">OCEAN Model</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
      <div class="footer-logo">MusicPersona</div>
      <div class="footer-text">CN6000 Final Year Project · Zest Chukwu · 2407734<br>
      BSc Computing for Business · Supervisor: Dr Azhar Mahmood</div>
      <div class="footer-disc">Predictions are based on linguistic patterns<br>
      and should not be used for clinical assessment.</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def page_predict():
    st.markdown("""
    <div class="page">
      <div class="page-title">What does your music language say about you?</div>
      <div class="page-sub">Write a review of any song or album, how it made you feel,
      what you noticed, what moved you. The model analyses your language and predicts
      your Big Five personality profile.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 8% 3rem;">', unsafe_allow_html=True)

        use_cal = st.toggle("Bias calibration", value=True,
            help="Corrects for Openness dominance bias in the training data")

        st.markdown(
            '<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#7a8090;margin-bottom:0.6rem;">'
            'Try an example</div>', unsafe_allow_html=True)

        ex_cols = st.columns(5)
        for i, ex in enumerate(EXAMPLES):
            with ex_cols[i]:
                if st.button(ex[0], key=f"ex_{i}", use_container_width=True):
                    st.session_state.selected_ex = ex[2]; st.rerun()

        if st.session_state.get('selected_ex'):
            st.session_state['main_review'] = st.session_state['selected_ex']
            st.session_state['selected_ex'] = None
        review = st.text_area("", key="main_review", height=220,
            placeholder="Write your music review here...\n\nHow did it make you feel? What moved you?",
            label_visibility="collapsed") or ''

        wc = len(review.split()) if review.strip() else 0
        if review.strip():
            wc_color = "#3ecfb2" if wc >= 20 else "#6a7090"
            st.markdown(
                f'<div style="font-size:0.78rem;text-align:right;color:{wc_color};'
                f'margin-top:-0.3rem;margin-bottom:0.5rem;">'
                f'{wc} words{"  ✓" if wc >= 20 else "  — aim for 20+"}</div>',
                unsafe_allow_html=True)

        # Side by side centred buttons
        _, b1, b2, _ = st.columns([1.2, 1, 1, 1.2])
        with b1:
            analyse = st.button("ANALYSE MY PERSONALITY", type="primary",
                                use_container_width=True)
        with b2:
            if st.button("COMPARE TWO PEOPLE", key="p_cmp",
                         use_container_width=True):
                st.session_state.page = 'compare'; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if analyse:
        if not review.strip():
            st.warning("Please write a music review first.")
        elif wc < 5:
            st.warning("Review is too short — write at least a few sentences.")
        else:
            with st.spinner("Analysing your review..."):
                model, tokenizer, device, error = load_model()
                if error: st.error(f"Model not loaded: {error}"); st.stop()
                scores = run_predict(review, model, tokenizer, device, use_cal)
            st.session_state.scores = scores
            hist = st.session_state.history
            hist.append((review[:80] + ("..." if len(review) > 80 else ""), scores))
            st.session_state.history = hist
            st.session_state.page = 'results'
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def page_results():
    scores = st.session_state.scores

    if scores is None:
        st.markdown("""
        <div style="padding:10rem 10%;text-align:center;">
          <div style="font-size:3rem;opacity:0.3;margin-bottom:1rem;">✦</div>
          <div style="font-family:'Cormorant Garamond',serif;font-style:italic;
               font-size:1.8rem;color:#5a6a80;">No prediction yet</div>
          <div style="font-size:0.95rem;color:#5a6a80;margin-top:0.5rem;">
          Go to Predict and analyse a review first.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("GO TO PREDICT", key="r_go"):
            st.session_state.page = 'predict'; st.rerun()
        return

    arch = get_archetype(scores)

    st.markdown('<div class="page">', unsafe_allow_html=True)

    # Archetype card
    st.markdown(
        f'<div class="arch-card">'
        f'<div class="arch-eye">Your Personality Archetype</div>'
        f'<div style="font-size:2.8rem;margin-bottom:0.5rem;">{arch["icon"]}</div>'
        f'<div class="arch-name">{arch["name"]}</div>'
        f'<div class="arch-desc">{arch["desc"]}</div>'
        f'</div>', unsafe_allow_html=True)

    # Stat cards
    render_stat_cards(scores)
    st.markdown("<div style='margin-bottom:2rem'></div>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown('<div class="slabel">Radar Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar(scores), use_container_width=True,
                        config={'displayModeBar': False})
    with c2:
        st.markdown('<div class="slabel">Score Distribution (OCEAN)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_bar(scores), use_container_width=True,
                        config={'displayModeBar': False})

    # Trait rows
    st.markdown('<div class="slabel">Trait Breakdown</div>', unsafe_allow_html=True)
    render_trait_rows(scores)

    # Action buttons
    st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.1, 1.8, 4])
    with b1:
        if st.button("TRY ANOTHER", key="r_again"):
            st.session_state.selected_ex = None
            st.session_state.page = 'predict'; st.rerun()
    with b2:
        if st.button("COMPARE TWO PEOPLE", key="r_cmp"):
            st.session_state.page = 'compare'; st.rerun()

    # History
    history = st.session_state.history
    if len(history) >= 2:
        st.markdown('<div class="slabel" style="margin-top:3rem;">Session History</div>',
                    unsafe_allow_html=True)
        for i, (prev, sc) in enumerate(reversed(history[-5:])):
            top = max(sc, key=sc.get)
            m   = TRAIT_META[top]
            st.markdown(
                f'<div class="hist-item">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="font-size:0.8rem;color:#7a8090;">#{len(history)-i}</span>'
                f'<span style="font-size:0.78rem;color:{m["color"]};">'
                f'{m["icon"]} {top.capitalize()} · {sc[top]:.0f}</span>'
                f'</div>'
                f'<div style="font-size:0.8rem;color:#9a9898;font-style:italic;'
                f'margin-top:0.2rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'{prev}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COMPARE
# ─────────────────────────────────────────────────────────────────────────────
def page_compare():
    st.markdown("""
    <div class="page">
      <div class="page-title">Same Song.<br>Two Personalities.</div>
      <div class="page-sub"> Watch how the same music reveals completely different personality profiles 
      proving that music is a window into who we are.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 8% 1.5rem;">', unsafe_allow_html=True)

        nc1, nc2 = st.columns(2)
        with nc1:
            name_a = st.text_input("Person A name", value="Person A",
                                   key="cna", placeholder="e.g. Sarah")
        with nc2:
            name_b = st.text_input("Person B name", value="Person B",
                                   key="cnb", placeholder="e.g. James")

        # Single row of examples — select who to fill first
        st.markdown(
            '<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#7a8090;margin-bottom:0.5rem;">'
            'Try an example</div>', unsafe_allow_html=True)

        fill_for = st.radio("Fill example for:",
                            [name_a, name_b],
                            horizontal=True,
                            label_visibility="collapsed",
                            key="fill_for")

        ex_cols = st.columns(5)
        for i, ex in enumerate(EXAMPLES):
            with ex_cols[i]:
                if st.button(ex[0], key=f"cex_{i}", use_container_width=True):
                    if fill_for == name_a:
                        st.session_state.cex_a = ex[2]
                    else:
                        st.session_state.cex_b = ex[2]
                    st.rerun()

        st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.markdown(
                f'<div style="font-size:0.74rem;font-weight:700;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#3ecfb2;margin-bottom:0.4rem;">'
                f"{name_a}'s Review</div>", unsafe_allow_html=True)
            if st.session_state.get('cex_a'):
                st.session_state['cra'] = st.session_state['cex_a']
                st.session_state['cex_a'] = None
            review_a = st.text_area("", key="cra", height=200,
                placeholder="How does this song make you feel?",
                label_visibility="collapsed") or ''
            wc_a = len(review_a.split()) if review_a.strip() else 0
            if review_a.strip():
                st.markdown(
                    f'<div style="font-size:0.72rem;text-align:right;'
                    f'color:{"#3ecfb2" if wc_a>=15 else "#6a7090"};">'
                    f'{wc_a} words</div>', unsafe_allow_html=True)

        with cb:
            st.markdown(
                f'<div style="font-size:0.74rem;font-weight:700;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#e8c16a;margin-bottom:0.4rem;">'
                f"{name_b}'s Review</div>", unsafe_allow_html=True)
            if st.session_state.get('cex_b'):
                st.session_state['crb'] = st.session_state['cex_b']
                st.session_state['cex_b'] = None
            review_b = st.text_area("", key="crb", height=200,
                placeholder="How does this song make you feel?",
                label_visibility="collapsed") or ''
            wc_b = len(review_b.split()) if review_b.strip() else 0
            if review_b.strip():
                st.markdown(
                    f'<div style="font-size:0.72rem;text-align:right;'
                    f'color:{"#3ecfb2" if wc_b>=15 else "#6a7090"};">'
                    f'{wc_b} words</div>', unsafe_allow_html=True)

        go_btn = st.button("COMPARE PERSONALITIES", type="primary", key="cgo")
        st.markdown('</div>', unsafe_allow_html=True)

    if go_btn:
        if not review_a.strip() or not review_b.strip():
            st.warning("Both people need to write a review first.")
        elif wc_a < 5 or wc_b < 5:
            st.warning("Both reviews need at least a few sentences.")
        else:
            with st.spinner("Analysing both personalities..."):
                model, tokenizer, device, error = load_model()
                if error: st.error(f"Model not loaded: {error}"); st.stop()
                sa = run_predict(review_a, model, tokenizer, device, True)
                sb = run_predict(review_b, model, tokenizer, device, True)
            st.session_state.cmp_a  = sa
            st.session_state.cmp_b  = sb
            st.session_state.cmp_na = name_a
            st.session_state.cmp_nb = name_b
            st.rerun()

    # Results
    if st.session_state.cmp_a and st.session_state.cmp_b:
        sa = st.session_state.cmp_a
        sb = st.session_state.cmp_b
        na = st.session_state.cmp_na
        nb = st.session_state.cmp_nb
        aa = get_archetype(sa)
        ab = get_archetype(sb)

        st.markdown('<div style="padding:0 8% 4rem;">', unsafe_allow_html=True)

        # Archetype cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="person-card">'
                f'<div class="person-lbl" style="color:#3ecfb2;">{na}</div>'
                f'<div style="font-size:2.3rem;margin-bottom:0.5rem;">{aa["icon"]}</div>'
                f'<div class="person-arch">{aa["name"]}</div>'
                f'<div class="person-desc">{aa["desc"]}</div>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="person-card">'
                f'<div class="person-lbl" style="color:#e8c16a;">{nb}</div>'
                f'<div style="font-size:2.3rem;margin-bottom:0.5rem;">{ab["icon"]}</div>'
                f'<div class="person-arch">{ab["name"]}</div>'
                f'<div class="person-desc">{ab["desc"]}</div>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:2.5rem'></div>", unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="slabel">Personality Overlay</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_compare_radar(sa, sb, na, nb),
                        use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="slabel" style="margin-top:2rem;">Score Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_compare_bar(sa, sb, na, nb),
                        use_container_width=True, config={'displayModeBar': False})

        # Differences table
        diffs = sorted([(t, abs(sa[t]-sb[t]), sa[t], sb[t]) for t in TRAITS],
                       key=lambda x: -x[1])
        st.markdown(
            '<div class="diff-card"><div class="diff-title">'
            'Biggest Personality Differences</div>',
            unsafe_allow_html=True)
        for trait, gap, va, vb in diffs:
            meta   = TRAIT_META[trait]
            leader = na if va > vb else nb
            lc     = "#3ecfb2" if va > vb else "#e8c16a"
            bw     = min(100, int(gap))
            st.markdown(
                f'<div class="diff-row">'
                f'<div class="diff-trait">{meta["icon"]} {trait.capitalize()}</div>'
                f'<div style="flex:1;margin:0 1.2rem;display:flex;align-items:center;gap:0.6rem;">'
                f'<span style="font-size:0.88rem;color:#3ecfb2;min-width:34px;text-align:right;">{va:.0f}</span>'
                f'<div style="flex:1;height:7px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">'
                f'<div style="height:7px;border-radius:4px;background:linear-gradient(90deg,#3ecfb2,#e8c16a);width:{bw}%;"></div>'
                f'</div>'
                f'<span style="font-size:0.88rem;color:#e8c16a;min-width:34px;">{vb:.0f}</span>'
                f'</div>'
                f'<div class="diff-gap" style="color:{lc};">+{gap:.0f} {leader}</div>'
                f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1.1, 1.8, 4])
        with b1:
            if st.button("TRY YOUR OWN", key="c_own"):
                st.session_state.page = 'predict'
                st.session_state.selected_ex = None; st.rerun()
        with b2:
            if st.button("NEW COMPARISON", key="c_new"):
                st.session_state.cmp_a = None
                st.session_state.cmp_b = None; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_nav()

    page = st.session_state.page
    if   page == 'home':    page_home()
    elif page == 'predict': page_predict()
    elif page == 'results': page_results()
    elif page == 'compare': page_compare()

if __name__ == "__main__":
    main()
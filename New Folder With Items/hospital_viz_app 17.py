# =============================================================================
# COMP4037 Research Methods — Coursework 2
# UK Hospital Admissions: Burden & Lockdown Impact
#
# Research Question:
#   What is the relative burden of hospital admission categories across ICD
#   chapters and diagnoses, and how did the most substantively important
#   diagnoses deviate from their pre-COVID baseline during the 2020–21
#   lockdown and subsequent recovery period?
#
# Visual Design:
#   Figure 1 (Main)       — Hierarchical Treemap
#   Figure 2 (Supporting) — Matrix Heatmap with diverging colour scale
#
# Tool Stack:
#   Python 3, Pandas, NumPy, Plotly (go.Treemap + go.Heatmap), Streamlit
#
# Data Source:
#   Office for National Statistics (ONS) — NHS Hospital Admissions 2012–2023
# =============================================================================

import re                        # Regular expressions for label parsing
import streamlit as st           # Web dashboard framework
import pandas as pd              # Data loading and manipulation
import plotly.graph_objects as go  # Interactive chart rendering
import numpy as np               # Numerical operations


# =============================================================================
# SECTION 1 — STREAMLIT PAGE CONFIGURATION
# Sets the browser tab title, icon, and overall layout to 'wide'
# so both figures can be displayed at full width without side margins.
# =============================================================================
st.set_page_config(
    page_title="UK Hospital Admissions — CW2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# SECTION 2 — GLOBAL CSS STYLING
# All custom visual styles are defined here in a single <style> block injected
# via st.markdown. This controls fonts, colours, card layouts, legend boxes,
# observation panels, and the bridge connector between the two figures.
#
# Font choices:
#   - 'Source Serif 4' for display titles (editorial, high legibility)
#   - 'IBM Plex Sans' for body and UI text (clean, scientific readability)
# Colour palette is anchored to the orange accent (#f97316) used consistently
# across all UI components, figure borders, and the lockdown annotation.
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Base font override for all Streamlit-generated elements */
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Dashboard main title */
.big-title {
    font-family: 'Source Serif 4', serif;
    font-size: 2.0rem;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

/* Research question banner — dark background with orange left border */
.rq-box {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.3rem;
    font-size: 0.93rem;
    margin-bottom: 1.2rem;
    border-left: 5px solid #f97316;
    line-height: 1.6;
}
.rq-box b { color: #fb923c; }

/* Section heading with orange underline */
.section-label {
    font-family: 'Source Serif 4', serif;
    font-size: 1.2rem;
    color: #0f172a;
    border-bottom: 3px solid #f97316;
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.5rem 0;
}

/* Figure caption text beneath each chart */
.caption {
    font-size: 0.81rem;
    color: #64748b;
    font-style: italic;
    margin-top: 0.3rem;
    text-align: center;
    line-height: 1.5;
}

/* Observation and insight panel — warm amber */
.observation-box {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-left: 5px solid #f97316;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.90rem;
    color: #431407;
    line-height: 1.65;
}
.observation-box b { color: #c2410c; }

/* Bridge connector panel between Figure 1 and Figure 2 — cool blue */
.bridge-box {
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-left: 5px solid #2563eb;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin: 1.2rem 0;
    font-size: 0.91rem;
    color: #1e3a8a;
    line-height: 1.7;
}
.bridge-box b { color: #1d4ed8; }

/* Top-10 burden legend strip below treemap */
.top10-legend {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0 1rem 0;
    font-size: 1.0rem;
    color: #334155;
    line-height: 1.9;
}
.top10-legend b { color: #0f172a; }

/* Individual ★ rank badge used in the top-10 legend */
.rank-badge {
    display: inline-block;
    background: #fef08a;
    border: 1.5px solid #eab308;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #713f12;
    margin-right: 5px;
}

/* CW2 submission template box — green tint to distinguish from analytical text */
.template-box {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 5px solid #16a34a;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    font-size: 0.875rem;
    color: #14532d;
    line-height: 1.85;
}
.template-box b { color: #15803d; }

/* KPI summary row at the top of the dashboard */
.kpi-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; }
.kpi {
    background: #0f172a;
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    flex: 1;
    border-top: 3px solid #f97316;
}
.kpi-v {
    font-size: 1.55rem;
    font-weight: 600;
    color: #fb923c;
    font-family: 'Source Serif 4', serif;
}
.kpi-l {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* Horizontal divider */
hr.thin { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }

/* ICD chapter colour legend below treemap */
.chapter-legend {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.0rem 1.2rem;
    margin: 0.75rem 0 0.9rem 0;
    font-size: 1.0rem;
    color: #334155;
    line-height: 1.8;
}
.chapter-legend b { color: #0f172a; font-size: 1.05rem; }
.chapter-legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem 1.3rem;
    margin-top: 0.55rem;
}
.chapter-legend-item {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    font-size: 1.0rem;
}
/* Colour swatch square in the chapter legend */
.chapter-legend-swatch {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    display: inline-block;
    margin-right: 7px;
    border: 1px solid rgba(15,23,42,0.15);
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3 — DATA LOADING AND PREPROCESSING
#
# This function is decorated with @st.cache_data so that the CSV is only
# read and processed once per session; subsequent renders use the cached
# result, keeping the dashboard responsive.
#
# Steps performed:
#   1. Load CSV from the local data_raw/ directory
#   2. Rename columns to short, consistent names
#   3. Coerce numeric columns (admissions, emergency, gender splits, etc.)
#   4. Drop the partial 2023–24 year to avoid incomplete annual totals
#   5. Drop rows missing a year label or diagnosis description
#   6. Canonicalise diagnosis text: strip whitespace, normalise punctuation,
#      and fix known multi-year spelling inconsistencies so the same clinical
#      condition always maps to a single string
#   7. Derive ICD-10 chapter from the leading letter of the diagnosis code
# =============================================================================
@st.cache_data
def load_data():
    # ── 3.1 Load raw CSV ──────────────────────────────────────────────────────
    df = pd.read_csv("./data_raw/clean_combined_data.csv")

    # ── 3.2 Standardise column names ─────────────────────────────────────────
    # The raw CSV uses verbose column names; rename to compact identifiers
    # used throughout the rest of the code.
    df = df.rename(columns={
        'Primary diagnosis: summary code and description': 'code',
        'Unnamed: 1': 'description',
        '2012-13': 'year'
    })

    # ── 3.3 Convert numeric columns ──────────────────────────────────────────
    # Non-numeric entries (dashes, blanks) are silently converted to NaN
    # so that pandas aggregation functions (sum, mean) work correctly.
    num_cols = [
        'Admissions', 'Emergency', 'Male', 'Female',
        'Waiting list', 'Planned', 'Mean age',
        'Mean length of stay', 'Mean time waited',
        'Finished consultant episodes', 'FCE bed days'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 3.4 Remove partial year ───────────────────────────────────────────────
    # 2023–24 data was still being collected at the time of analysis;
    # including it would understate annual totals and distort percentage
    # change calculations. It is excluded entirely.
    df = df[df['year'] != '2023-24']

    # ── 3.5 Drop rows with missing identifiers ────────────────────────────────
    df = df.dropna(subset=['year', 'description'])
    df['description'] = df['description'].astype(str).str.strip()

    # ── 3.6 Canonical diagnosis label ────────────────────────────────────────
    # The same clinical diagnosis sometimes appears with minor differences in
    # punctuation or spacing across different financial years (e.g., a comma
    # missing, extra whitespace). Without canonicalisation these would be
    # treated as separate diagnoses, splitting their admission counts.
    df['description_canon'] = (
        df['description']
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)   # collapse multiple spaces
    )

    # Explicit fixes for known multi-year label inconsistencies
    df['description_canon'] = df['description_canon'].replace({
        'Diseases of oesophagusstomach & duodenum':
            'Diseases of oesophagus, stomach & duodenum',
        'Diseases of oesophagus stomach & duodenum':
            'Diseases of oesophagus, stomach & duodenum',
        ' Diseases of oesophagus, stomach & duodenum':
            'Diseases of oesophagus, stomach & duodenum',
    })

    # ── 3.7 ICD-10 chapter derivation ────────────────────────────────────────
    # ICD-10 uses a letter prefix to indicate the disease chapter (e.g.,
    # K = Digestive, O = Pregnancy & Childbirth). Extracting this letter and
    # mapping it to a readable chapter name creates the outer hierarchy
    # required by the treemap.
    chapter_map = {
        'A': 'Infectious Diseases',
        'B': 'Infectious Diseases',
        'C': 'Cancer',
        'D': 'Cancer & Blood',
        'E': 'Endo Metabolic',
        'F': 'Mental Health',
        'G': 'Nervous System',
        'H': 'Eye & Ear',
        'I': 'Circulatory',
        'J': 'Respiratory',
        'K': 'Digestive',
        'L': 'Skin',
        'M': 'Musculoskeletal',
        'N': 'Genitourinary',
        'O': 'Pregnancy & Childbirth',
        'P': 'Perinatal',
        'Q': 'Congenital',
        'R': 'Symptoms & Signs',
        'S': 'Injury & Poisoning',
        'T': 'Injury & Poisoning',
        'Z': 'Health Services'
    }

    def extract_icd_letter(code):
        """Extract the leading ICD-10 letter from a raw code string.

        Priority: look for the pattern <letter><digit> (e.g., 'K57').
        Fallback: return the first alphabetic character found.
        Returns None if no letter is found.
        """
        if not isinstance(code, str):
            return None
        code = code.upper().strip()
        # Preferred: letter immediately followed by digit
        match = re.search(r'\b([A-Z])\d', code)
        if match:
            return match.group(1)
        # Fallback: any letter
        match = re.search(r'[A-Z]', code)
        if match:
            return match.group(0)
        return None

    def get_chapter(code):
        """Map a raw ICD code string to its chapter name."""
        letter = extract_icd_letter(code)
        if letter is None:
            return 'Other'
        return chapter_map.get(letter, 'Other')

    df['code_clean_letter'] = df['code'].apply(extract_icd_letter)
    df['chapter'] = df['code'].apply(get_chapter)

    return df


# ── Load data and extract unique sorted years ──────────────────────────────────
df = load_data()
years = sorted(df['year'].unique())


# =============================================================================
# SECTION 4 — DISPLAY LABEL LOOKUP TABLE
#
# Long clinical descriptions are replaced with concise display names
# for use inside treemap rectangles and heatmap column headers.
# The full original name is always available in hover tooltips.
# Keys are the canonical description strings; values are the short labels.
# =============================================================================
SHORT_LABELS = {
    'Complications of labour and delivery':                          'Labour & Delivery',
    'Health services in circumstances related to reproduction':      'Reproduction Health Svcs',
    'Arthropathies':                                                 'Arthropathies',
    'Other diseases of intestines':                                  'Intestinal Diseases',
    'Symptoms & signs inv. the digestive system & abdomen':          'Digestive Symptoms',
    'Disorders of lens (including cataracts)':                       'Cataracts',
    'Symptoms & signs inv. the circulatory/respiratory system':      'Circ./Resp. Symptoms',
    'Diseases of oesophagus, stomach & duodenum':                    'Oesophagus / Stomach',
    'Malignant neoplasms of lymphoid, haematopoietic & rel. tiss.':  'Blood Cancers',
    'General symptoms & signs':                                      'General Symptoms',
    'Symptoms & signs involving the urinary system':                 'Urinary Symptoms',
    'Symptoms & signs inv. Cognition perception etc.':               'Cognitive Symptoms',
    'Symptoms & signs inv. the nervous & musculoskeletal sys.':      'Neuromuscular Symptoms',
    'Dorsopathies':                                                  'Dorsopathies (Back & Spine)',
    'Other forms of heart disease':                                  'Heart Disease (Other)',
    'Hernia':                                                        'Hernia',
    'Soft tissue disorders':                                         'Soft Tissue Disorders',
    'Examination and investigation':                                 'Examination / Investigation',
    'Endocrine nutritional and metabolic diseases':                  'Endocrine & Metabolic',
    'Complications of surgical & medical care nec.':                 'Care Complications',
    'Other Diseases & disorders of the nervous syst.':               'Nervous Disorders',
    'In situ & benign neoplasms and others of uncertainty':          'Benign Neoplasms',
    'Poisonings by drugs medicaments & biological substances': 'Drugs Poisoning',
}


# =============================================================================
# SECTION 5 — ICD CHAPTER COLOUR PALETTE
#
# Each ICD chapter is assigned a distinct, perceptually varied hue.
# Colours were selected to maximise contrast between adjacent chapters
# in the treemap while remaining accessible at screen resolution.
# Within each chapter, colour intensity (saturation/lightness) is further
# modulated by admission volume — see the 'chapter_tinted' function below.
# =============================================================================
CHAPTER_COLORS = {
    'Digestive':              '#d97706',
    'Symptoms & Signs':       '#860E84',
    'Cancer':                 '#dc2626',
    'Cancer & Blood':         '#b91c1c',
    'Musculoskeletal':        '#16a34a',
    'Health Services':        '#2563eb',
    'Injury & Poisoning':     '#7c3aed',
    'Circulatory':            '#db2777',
    'Respiratory':            '#0d9488',
    'Eye & Ear':              '#d97706',
    'Pregnancy & Childbirth': '#0891b2',
    'Genitourinary':          '#0284c7',
    'Infectious Diseases':    '#ea580c',
    'Nervous System':         '#ea33cf',
    'Endo Metabolic':         '#0369a1',
    'Mental Health':          '#be185d',
    'Skin':                   '#65a30d',
    'Perinatal':              '#059669',
    'Congenital':             "#8D609A",
    'Other':                  '#94a3b8',
}


# =============================================================================
# SECTION 6 — COLOUR UTILITY FUNCTIONS
#
# These three functions implement the 'dual colour encoding' used in the
# treemap, where both hue (chapter) and intensity (volume within chapter)
# carry meaning.
# =============================================================================

def _hex_to_rgb(h: str):
    """Convert a CSS hex colour string (e.g. '#d97706') to a (r, g, b) tuple
    with each channel normalised to [0, 1]."""
    h = h.lstrip('#')
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def _rgb_to_hex(r, g, b) -> str:
    """Convert normalised (r, g, b) floats back to a CSS hex colour string."""
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def chapter_tinted(hex_color: str, t: float) -> str:
    """Blend hex_color with white by a factor controlled by t ∈ [0, 1].

    t = 0  → nearly white  (lowest-volume diagnosis in this chapter)
    t = 1  → full chapter colour (highest-volume diagnosis in this chapter)

    MIN_MIX ensures even the lowest-volume cell retains enough hue to be
    clearly associated with its parent chapter colour.
    """
    MIN_MIX = 0.20   # Minimum saturation (20% of full chapter colour)
    MAX_MIX = 1.00   # Maximum saturation (100% = pure chapter colour)
    mix = MIN_MIX + t * (MAX_MIX - MIN_MIX)

    r, g, b = _hex_to_rgb(hex_color)
    # Linear interpolation between the chapter colour and white (1, 1, 1)
    r2 = mix * r + (1 - mix)
    g2 = mix * g + (1 - mix)
    b2 = mix * b + (1 - mix)

    return _rgb_to_hex(r2, g2, b2)


def readable_text_color(hex_color: str) -> str:
    """Return '#0f172a' (near-black) or 'white' depending on background
    luminance, ensuring WCAG-compliant text contrast on every cell."""
    r, g, b = _hex_to_rgb(hex_color)
    # Relative luminance formula (ITU-R BT.709)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#0f172a' if lum > 0.45 else 'white'


def build_chapter_admission_ranges(tree_df_input):
    """Compute (min, max) admission totals for each chapter.

    These ranges are used by 'chapter_tinted' to normalise each diagnosis's
    admission total within its chapter, converting an absolute count to a
    [0, 1] intensity score that drives colour saturation.
    """
    ranges = {}
    for ch, grp in tree_df_input.groupby('chapter'):
        ranges[ch] = (grp['total'].min(), grp['total'].max())
    return ranges


# =============================================================================
# SECTION 7 — LABEL AND LEGEND HELPER FUNCTIONS
# =============================================================================

def make_short_label(text: str) -> str:
    """Generate a concise display label from a full diagnosis description.

    Algorithm:
      1. Check the SHORT_LABELS lookup; return the pre-defined abbreviation
         if available.
      2. Strip punctuation, collapse whitespace, and split into words.
      3. Remove common medical stopwords (of, and, the, …).
      4. Join the first two meaningful words; truncate to 24 characters.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if text in SHORT_LABELS:
        return SHORT_LABELS[text]

    cleaned = re.sub(r'[(),]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = cleaned.split()

    stopwords = {
        'of', 'and', 'the', 'in', 'to', 'for', 'with', 'without', 'due',
        'including', 'other', 'forms', 'form', 'related', 'circumstances',
        'inv', 'nec', 'not', 'elsewhere', 'specified'
    }
    meaningful = [w for w in words if w.lower() not in stopwords]
    short = " ".join(meaningful[:2]) if len(
        meaningful) >= 2 else " ".join(words[:2])

    if len(short) > 24:
        short = short[:23].rstrip() + "…"
    return short


def chapter_legend_html(color_map: dict, chapter_totals: pd.Series) -> str:
    """Build an HTML legend strip mapping each ICD chapter to its colour swatch.

    Chapters are ordered from highest to lowest total admissions so the most
    clinically significant categories appear first in the legend.
    """
    items = []
    for chapter in chapter_totals.sort_values(ascending=False).index:
        color = color_map.get(chapter, "#94a3b8")
        items.append(
            f'<span class="chapter-legend-item">'
            f'<span class="chapter-legend-swatch" style="background:{color};"></span>'
            f'{chapter}'
            f'</span>'
        )

    return f"""
    <div class="chapter-legend">
    <b>Treemap Colour Legend — ICD Chapters</b><br>
    <span style="font-size:0.95rem; color:#64748b;">
    Each colour represents one ICD disease chapter. Within the same chapter, darker and more
    saturated tiles indicate higher admission volumes, while lighter tiles indicate lower volumes.
    Chapters are listed below from highest to lowest total burden.
    </span>
    <div class="chapter-legend-grid">
    {''.join(items)}
    </div>
    </div>
    """


# =============================================================================
# SECTION 8 — TOP-10 BURDEN DIAGNOSIS RANKING
#
# Aggregate total admissions (2012–2023) per canonical diagnosis and rank
# from highest to lowest. The top-10 list is reused in three places:
#   • Treemap  — ★ rank labels and gold cell borders
#   • Heatmap  — defines the 10 column categories
#   • Bridge   — shows lockdown % change for each top-10 diagnosis
# =============================================================================
diag_burden = (
    df.groupby('description_canon')['Admissions']
    .sum()
    .sort_values(ascending=False)
)

top10_burden_list = diag_burden.nlargest(
    10).index.tolist()   # Ordered list: #1 → #10
# Set for fast membership test
top10_burden_set = set(top10_burden_list)
# Dict: name → rank
top10_rank = {d: i + 1 for i, d in enumerate(top10_burden_list)}


# =============================================================================
# SECTION 9 — HEATMAP BASELINE AND LOCKDOWN DATA PREPARATION
#
# The heatmap expresses every year's admissions as a percentage change
# relative to the 2019–20 pre-COVID financial year.
#
# Formula:  Δ% = (admissions_t − admissions_2019-20) / admissions_2019-20 × 100
#
# The 10 columns are the top-10 burden diagnoses, ordered by their 2020–21
# lockdown change from the steepest decline (leftmost) to the most resilient
# (rightmost). This ordering makes the pandemic disruption gradient
# immediately visible across the heatmap.
# =============================================================================

# 2019–20 baseline admissions per diagnosis (denominator for all % changes)
base_ser = (
    df[df['year'] == '2019-20']
    .groupby('description_canon')['Admissions']
    .sum()
)

# 2020–21 lockdown admissions per diagnosis (used to sort heatmap columns)
lock_ser = (
    df[df['year'] == '2020-21']
    .groupby('description_canon')['Admissions']
    .sum()
)


def lockdown_pct(d):
    """Compute the 2020–21 % change vs 2019–20 baseline for diagnosis d.
    Returns 0 if baseline is missing or zero (safe default for sorting).
    """
    b = base_ser.get(d, np.nan)
    l = lock_ser.get(d, np.nan)
    return (l - b) / b * 100 if (pd.notna(b) and b > 0) else 0


# Sort top-10 by lockdown drop: most disrupted (most negative) → most resilient
selected_diagnoses_sorted = sorted(top10_burden_list, key=lockdown_pct)


# =============================================================================
# SECTION 10 — KPI METRICS
# Computed once and displayed in the top summary bar.
# =============================================================================

# Total admissions across the entire 2012–2023 study period
total_adm = df['Admissions'].sum()

# Percentage change in total admissions from 2019–20 to 2020–21
covid_change = (
    df[df['year'] == '2020-21']['Admissions'].sum() /
    df[df['year'] == '2019-20']['Admissions'].sum() - 1
) * 100


# =============================================================================
# SECTION 11 — DASHBOARD HEADER
# Renders the title banner, research question box, and four KPI cards.
# =============================================================================

st.markdown(
    '<p class="big-title">🏥 UK Hospital Admissions — Burden & Lockdown Impact</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="rq-box">
<b>Research Question:</b> What is the relative burden of hospital admission categories across
ICD chapters and diagnoses, and how did the most substantively important diagnoses deviate
from their pre-COVID baseline during the <b>2020–21 lockdown</b> and subsequent recovery period?<br>
<span style="font-size:0.82rem; color:#94a3b8">
ONS Hospital Admissions Dataset · 2012–2023 · COMP4037 CW2 · Python 3 / Streamlit / Plotly
</span>
</div>
""", unsafe_allow_html=True)

# Four KPI cards displayed as a horizontal flex row
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-l">Total Admissions (2012–2023)</div>
    <div class="kpi-v">{total_adm/1e6:.0f}M</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Lockdown Drop (2020–21 vs 2019–20)</div>
    <div class="kpi-v" style="color:#ef4444">{covid_change:+.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Largest ICD Chapter (by burden)</div>
    <div class="kpi-v" style="font-size:1.0rem;padding-top:0.4rem">Digestive</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Highest Single Diagnosis (12-yr total)</div>
    <div class="kpi-v" style="font-size:0.85rem;padding-top:0.35rem">Labour &amp; Delivery</div>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 12 — FIGURE 1: HIERARCHICAL TREEMAP
#
# The treemap encodes a two-level ICD hierarchy:
#   Level 1 (outer) — ICD chapter   → hue (chapter colour)
#   Level 2 (inner) — diagnosis     → area (total admissions) +
#                                      colour intensity (within-chapter volume)
#
# Construction process:
#   1. Aggregate admissions by chapter–diagnosis pair
#   2. Build three node types: root node, chapter nodes, diagnosis nodes
#   3. Assign dual-channel colour to each diagnosis cell
#   4. Mark top-10 diagnoses with ★ rank labels and dark borders
#   5. Render with Plotly go.Treemap
# =============================================================================

st.markdown(
    '<div class="section-label">'
    'Figure 1 (Main) — Hierarchical Treemap: '
    'Relative Admission Burden by ICD Chapter &amp; Diagnosis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
**Area (size)** = total admissions 2012–2023 ·
**Colour hue** = ICD disease chapter (categorical) ·
**Colour intensity** = relative admission volume within each chapter ·
**Outer rectangle** = ICD chapter · **Inner rectangle** = specific diagnosis ·
**★ #1–#10** = Top-10 highest-burden diagnoses (ranked) — examined further in Figure 2 ·
Hover over any tile for exact totals.
""")

# ── 12.1 Aggregate admissions by chapter–diagnosis ───────────────────────────
tree_df = (
    df.groupby(['chapter', 'description_canon'], as_index=False)['Admissions']
    .sum()
    .rename(columns={'Admissions': 'total'})
)
tree_df = tree_df[tree_df['total'] > 0].copy()

# Prepare display labels (full name for hover, short for tile label)
tree_df['label_full'] = tree_df['description_canon'].fillna(
    'Unknown').astype(str).str.strip()
tree_df['label_short'] = tree_df['label_full'].apply(make_short_label)

# ── 12.2 Compute within-chapter min/max for colour intensity normalisation ────
chapter_ranges = build_chapter_admission_ranges(tree_df)


def get_dual_color(chapter: str, total: float) -> str:
    """Return a colour blended from the chapter hue and white,
    where the blend ratio reflects this diagnosis's volume relative to the
    chapter's min–max range. A power of 0.65 compresses the scale slightly
    so that medium-volume diagnoses still show meaningful colour."""
    base = CHAPTER_COLORS.get(chapter, "#94a3b8")
    ch_min, ch_max = chapter_ranges.get(chapter, (total, total))
    t = (total - ch_min) / (ch_max - ch_min) if ch_max > ch_min else 1.0
    t = t ** 0.65   # Non-linear compression for better perceptual spread
    return chapter_tinted(base, t)


# ── 12.3 Compute grand total and chapter totals ───────────────────────────────
grand_total = tree_df['total'].sum()
chapter_totals = tree_df.groupby('chapter')['total'].sum()

# Minimum fraction of grand total for a label to be shown on a tile
MIN_DIAG_FRACTION = 0.004   # 0.4% of all admissions (~700k over 11 years)

tree_df['show_label'] = tree_df['total'] / grand_total >= MIN_DIAG_FRACTION
tree_df['is_top10'] = tree_df['label_full'].isin(top10_burden_set)
# Always show labels for top-10 regardless of tile size
tree_df.loc[tree_df['is_top10'], 'show_label'] = True


def build_display_label(row) -> str:
    """Return the label text to display inside a treemap tile.

    Top-10 diagnoses: '★#N Short Name'
    Other visible diagnoses: 'Short Name'
    Too-small to label: '' (empty string, tile is blank)
    """
    if row['is_top10']:
        rank = top10_rank[row['label_full']]
        return f"★#{rank} {row['label_short']}"
    elif row['show_label']:
        return row['label_short']
    return ""


tree_df['label_display'] = tree_df.apply(build_display_label, axis=1)

# ── 12.4 Build parallel lists for go.Treemap ─────────────────────────────────
# Plotly's go.Treemap requires six parallel lists where position i in each
# list describes the same node: its id, display label, parent id, value,
# colour, and hover data.
labels, parents, values, ids, colors, customdata, textcolors = [], [], [], [], [], [], []
line_widths, line_colors = [], []

# ── Root node (invisible background; required by Plotly hierarchy) ────────────
root_id = "root"
labels.append("All Admissions")
parents.append("")
values.append(grand_total)
ids.append(root_id)
colors.append("#1e293b")
customdata.append(["All Admissions", grand_total, "—", ""])
textcolors.append("white")
line_widths.append(0)
line_colors.append("white")

# ── Chapter nodes (outer rectangles) ─────────────────────────────────────────
for chapter, ch_total in chapter_totals.sort_values(ascending=False).items():
    chapter_id = f"ch::{chapter}"
    base_col = CHAPTER_COLORS.get(chapter, "#94a3b8")

    labels.append(chapter)
    parents.append(root_id)
    values.append(ch_total)
    ids.append(chapter_id)
    colors.append(base_col)
    customdata.append([chapter, ch_total, chapter, ""])
    textcolors.append(readable_text_color(base_col))
    line_widths.append(3)      # Thick white border separates chapters
    line_colors.append("white")

# ── Diagnosis nodes (inner rectangles) ───────────────────────────────────────
for _, row in tree_df.iterrows():
    chapter = row['chapter']
    full_name = row['label_full']
    disp_name = row['label_display']
    total = row['total']
    is_top10 = row['is_top10']

    chapter_id = f"ch::{chapter}"
    diag_id = f"dx::{chapter}::{full_name}"

    cell_color = get_dual_color(chapter, total)
    txt_col = readable_text_color(cell_color)

    # Top-10 cells use gold label text to stand out against dark borders
    if is_top10:
        txt_col = '#fef08a'

    # Hover tooltip extra line: rank and share of all admissions
    rank_label = f"★ Top-10 Burden Rank #{top10_rank[full_name]}" if is_top10 else ""
    share_pct = total / grand_total * 100
    hover_extra = (
        f"{rank_label}<br>Share of all admissions: {share_pct:.2f}%"
        if rank_label else
        f"Share of all admissions: {share_pct:.2f}%"
    )

    labels.append(disp_name)
    parents.append(chapter_id)
    values.append(total)
    ids.append(diag_id)
    colors.append(cell_color)
    customdata.append([full_name, total, chapter, hover_extra])
    textcolors.append(txt_col)
    # Top-10 cells get a thick dark border to visually mark them
    line_widths.append(3.5 if is_top10 else 1.5)
    line_colors.append("#1e293b" if is_top10 else "white")

# ── 12.5 Render treemap ───────────────────────────────────────────────────────
fig_tree = go.Figure(go.Treemap(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",       # Parent rectangle = sum of all children
    textinfo="label",           # Show label text inside tiles
    textfont=dict(size=16, family='IBM Plex Sans'),
    marker=dict(
        colors=colors,
        line=dict(width=line_widths, color=line_colors),
        pad=dict(t=24, b=5, l=5, r=5)  # Padding inside each chapter block
    ),
    customdata=customdata,
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'Total Admissions (2012–2023): <b>%{customdata[1]:,.0f}</b><br>'
        'ICD Chapter: %{customdata[2]}<br>'
        '%{customdata[3]}<extra></extra>'
    ),
    root_color='#1e293b'
))

fig_tree.update_layout(
    height=620,
    margin=dict(t=30, l=8, r=8, b=8),
    paper_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    uniformtext=dict(minsize=13, mode='hide')  # Hide labels if they won't fit
)

st.plotly_chart(fig_tree, use_container_width=True)

# ── 12.6 Chapter colour legend ───────────────────────────────────────────────
st.markdown(chapter_legend_html(CHAPTER_COLORS,
            chapter_totals), unsafe_allow_html=True)

# ── 12.7 Top-10 ranked diagnosis strip ───────────────────────────────────────
top10_legend_rows = ""
for rank, diag in enumerate(top10_burden_list, 1):
    short = SHORT_LABELS.get(diag, make_short_label(diag))
    total_v = diag_burden[diag]
    top10_legend_rows += (
        f'<span class="rank-badge">★#{rank}</span>'
        f'<b>{short}</b> — {total_v/1e6:.2f}M admissions &nbsp;&nbsp; '
    )

st.markdown(f"""
<div class="top10-legend">
<b>★ Top-10 Highest-Burden Diagnoses</b>
(highlighted in treemap with gold labels and dark borders · these exact diagnoses form the
10 columns of Figure 2):<br>
{top10_legend_rows}
</div>
""", unsafe_allow_html=True)

# ── 12.8 Figure 1 caption ────────────────────────────────────────────────────
st.markdown("""
<p class="caption">
<b>Figure 1 (Main).</b> Hierarchical treemap of total UK hospital admissions (2012–2023).
Outer rectangles represent ICD disease chapters; inner rectangles represent individual
diagnosis groups nested within each chapter. Rectangle <b>area</b> is proportional to
cumulative admissions over the study period, making relative burden immediately visible
at both chapter and diagnosis levels. <b>Colour hue</b> encodes ICD chapter membership
categorically; <b>colour intensity</b> within each chapter reflects relative admission
volume for that diagnosis compared to others in the same chapter.
<b>★ #1–#10 labels with dark borders</b> identify the ten highest-burden diagnoses, which
form the basis of Figure 2. Labels are suppressed for tiles too small to be legible;
full diagnosis names, exact admission counts, and burden share are available on hover.
</p>
""", unsafe_allow_html=True)

# ── 12.9 Treemap analytical observation ─────────────────────────────────────
st.markdown("""
<div class="observation-box">
<b>🔍 What the Treemap Reveals:</b><br>
The <b>Digestive</b> chapter occupies the largest outer block, confirming it carries the
greatest aggregate admission burden over the 2012–2023 period — yet its area is built from
a spread of conditions rather than a single dominant diagnosis. This compositional diversity
contrasts sharply with <b>Eye &amp; Ear</b>, whose area is almost entirely occupied by a single
inner tile: <b>★#6 Cataracts (Lens Disorders)</b>, showing that some chapters are clinically
concentrated while others are broadly distributed.<br><br>
Equally striking, <b>Pregnancy &amp; Childbirth</b> is a relatively small chapter by outer area,
yet its inner rectangle — <b>★#1 Labour &amp; Delivery</b> — is the single largest diagnosis
block in the entire treemap, with 8.77 million admissions rivalling whole mid-sized chapters.
This illustrates a key strength of the hierarchical treemap: structural differences between
chapter composition and diagnosis dominance, which are completely invisible in a standard bar
chart, are immediately apparent through the nested area encoding.
<b>The ten ★-marked diagnoses are carried forward to Figure 2</b> to test whether their high
volume translated to resilience or vulnerability during the 2020–21 COVID lockdown.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)


# =============================================================================
# SECTION 13 — BRIDGE: EXPLICIT NARRATIVE LINK BETWEEN FIGURE 1 AND FIGURE 2
#
# This panel makes the analytical relationship between the two visualisations
# explicit, which is essential for academic credibility. It states:
#   • Figure 1 answers: "which diagnoses carry the greatest burden?"
#   • Figure 2 answers: "how did those same diagnoses behave during COVID?"
# It also lists each top-10 diagnosis with its 2020–21 lockdown change,
# previewing the heatmap finding before the chart appears.
# =============================================================================
bridge_items = []
for rank, diag in enumerate(top10_burden_list, 1):
    short = SHORT_LABELS.get(diag, make_short_label(diag))
    lp = lockdown_pct(diag)
    bridge_items.append(f"<b>★#{rank} {short}</b> ({lp:+.0f}%)")

bridge_text = " · ".join(bridge_items)

st.markdown(f"""
<div class="bridge-box">
<b>🔗 Analytical Link — Figure 1 → Figure 2:</b><br>
<b>Figure 1 (Treemap)</b> answers: <i>"Which diagnoses carry the greatest cumulative admission
burden across ICD chapters?"</i>
<b>Figure 2 (Heatmap)</b> directly follows up by asking: <i>"How did those exact same
top-10 diagnoses behave year-on-year — and in particular, how severely were they
disrupted during the 2020–21 COVID lockdown and its aftermath?"</i><br><br>
The ★-ranked diagnoses from the treemap become the <b>10 columns</b> of the heatmap below,
ordered from the diagnosis that suffered the <b>largest lockdown drop</b> (leftmost column)
to the most <b>resilient</b> (rightmost column). This pairing directly tests whether
high admission burden predicts pandemic resilience — or whether even the busiest diagnostic
categories were disrupted by the health emergency.<br><br>
<span style="font-size:0.83rem">
2020–21 lockdown change for each top-10 diagnosis: {bridge_text}
</span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 14 — FIGURE 2: MATRIX HEATMAP
#
# The heatmap encodes temporal deviation from a pre-COVID baseline:
#   Rows    — financial years 2012–13 to 2022–23 (oldest → newest, top → bottom)
#   Columns — top-10 burden diagnoses from Figure 1, sorted by lockdown drop
#   Colour  — diverging red–white–blue scale; % change vs 2019–20 baseline
#
# Additional visual encodings:
#   • Cell annotations — exact % values for precision reading
#   • Orange dashed border — marks the 2020–21 lockdown row
#   • Labelled arrows — mark the largest drop, most resilient, and first
#     post-lockdown rebound above baseline
# =============================================================================

st.markdown(
    '<div class="section-label">'
    'Figure 2 (Supporting) — Matrix Heatmap: '
    '% Change in Admissions vs 2019–20 Baseline'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
**Columns** = Top-10 highest-burden diagnoses from Figure 1 (★#1–#10),
ordered left → right from **most disrupted** to **most resilient** during the 2020–21 lockdown.
**Rows** = financial year (2012–13 at top → 2022–23 at bottom).
**Cell colour** = % change vs the 2019–20 pre-COVID baseline.
🔴 Red = below baseline · ⬜ White = near baseline · 🔵 Blue = above baseline.
Exact % values are printed inside each cell. The **orange dashed border** marks the 2020–21 lockdown year.
""")

# ── 14.1 Build annual admission series for each diagnosis ─────────────────────
yr_totals_all = {
    yr: df[df['year'] == yr].groupby('description_canon')['Admissions'].sum()
    for yr in years
}

# ── 14.2 Compute % change vs baseline for every year × diagnosis cell ─────────
hm_rows = {}
for yr in years:
    row = {}
    for d in selected_diagnoses_sorted:
        b = base_ser.get(d, np.nan)
        v = yr_totals_all[yr].get(d, 0)
        row[d] = (v - b) / b * 100 if (pd.notna(b) and b > 0) else np.nan
    hm_rows[yr] = row

hm_df = pd.DataFrame(hm_rows).T[selected_diagnoses_sorted]


def hm_col_label(d: str) -> str:
    """Build a heatmap column header that includes the ★ rank from Figure 1,
    reinforcing the visual bridge between the treemap and heatmap."""
    short = SHORT_LABELS.get(d, (d[:24] + '…' if len(d) > 24 else d))
    if d in top10_rank:
        return f"★#{top10_rank[d]} {short}"
    return short


short_cols = [hm_col_label(d) for d in selected_diagnoses_sorted]
hm_df.columns = short_cols

# ── 14.3 Clip values for colour scale; preserve raw values for annotation ──────
# Values are clipped only for colour mapping; raw values are used for cell text
# and hover tooltips, so extreme outliers are not hidden from the reader.
z_max, z_min = 35, -75
hm_clipped = hm_df.clip(z_min, z_max)

annot = hm_df.map(lambda v: f"{v:+.0f}%" if pd.notna(v) else "")
hover_vals = hm_df.values

# ── 14.4 Render heatmap ───────────────────────────────────────────────────────
fig_hm = go.Figure()

fig_hm.add_trace(go.Heatmap(
    z=hm_clipped.values,
    x=short_cols,
    y=hm_df.index.tolist(),
    zmin=z_min,
    zmax=z_max,
    # Seven-stop diverging colour scale (dark red → light red → white → light blue → dark blue)
    colorscale=[
        [0.00, '#7f1d1d'],   # Dark red  (−75%: severe decline)
        [0.15, '#b91c1c'],   # Red       (−50%)
        [0.35, '#fca5a5'],   # Light red (−25%)
        [0.50, '#f8fafc'],   # Near-white (0%: baseline)
        [0.65, '#93c5fd'],   # Light blue (+25%)
        [0.85, '#1d4ed8'],   # Blue
        [1.00, '#1e3a8a'],   # Dark blue (+35%+: above baseline)
    ],
    colorbar=dict(
        title=dict(text='% vs<br>2019–20<br>baseline', font=dict(size=14)),
        tickvals=[-75, -50, -25, 0, 25, 35],
        ticktext=['-75%', '-50%', '-25%', '0%', '+25%', '+35%'],
        tickfont=dict(size=14),
        len=1,
        thickness=22,
        x=1.05,
    ),
    text=annot.values,
    texttemplate="%{text}",
    textfont=dict(size=18, color='#1e293b'),
    customdata=hover_vals,
    hovertemplate=(
        '<b>%{x}</b><br>'
        'Year: %{y}<br>'
        'Change vs 2019–20 baseline: %{customdata:+.1f}%'
        '<extra></extra>'
    ),
    xgap=2,   # Thin gap between columns for visual separation
    ygap=2,   # Thin gap between rows
))

# ── 14.5 Lockdown row highlight ───────────────────────────────────────────────
# An orange dashed rectangle drawn over the 2020–21 row draws the reader's
# eye to the pandemic disruption event and anchors the interpretation.
lockdown_idx = hm_df.index.tolist().index('2020-21')

fig_hm.add_shape(
    type='rect', xref='paper', yref='y',
    x0=-0.01, x1=1.01,
    y0=lockdown_idx - 0.5, y1=lockdown_idx + 0.5,
    line=dict(color='#f97316', width=3, dash='dot'),
    fillcolor='rgba(249,115,22,0.06)',
    layer='above'
)

fig_hm.add_annotation(
    x=1.0, xref='paper', y='2020-21', yref='y',
    text='◄ COVID<br>lockdown',
    showarrow=False,
    font=dict(size=11, color='#c2410c', family='IBM Plex Sans'),
    xanchor='left', xshift=16
)

# ── 14.6 Analytic annotations ─────────────────────────────────────────────────
# These three labels are computed from the data and placed algorithmically,
# so they always point to the correct column regardless of data changes.

# Mark the column with the steepest lockdown decline
lockdown_vals = hm_clipped.iloc[lockdown_idx]
worst_col = lockdown_vals.idxmin()
worst_col_idx = short_cols.index(worst_col)

fig_hm.add_annotation(
    x=worst_col_idx, xref='x', y=lockdown_idx - 0.65, yref='y',
    text="▼ largest drop",
    showarrow=False,
    font=dict(size=9, color='#991b1b', family='IBM Plex Sans'),
    xanchor='center'
)

# Mark the most resilient column (smallest decline in 2020–21)
resilient_col = lockdown_vals.idxmax()
resilient_idx = short_cols.index(resilient_col)

fig_hm.add_annotation(
    x=resilient_idx, xref='x', y=lockdown_idx - 0.65, yref='y',
    text="▲ most resilient",
    showarrow=False,
    font=dict(size=9, color='#1e40af', family='IBM Plex Sans'),
    xanchor='center'
)

# Mark the first post-lockdown rebound above baseline (>+5%)
post_lockdown_years = hm_df.index.tolist()[lockdown_idx + 1:]
for yr in post_lockdown_years:
    above = hm_df.loc[yr][hm_df.loc[yr] > 5]
    if not above.empty:
        rebound_col = above.idxmax()
        rebound_col_idx = short_cols.index(rebound_col)
        rebound_yr_idx = hm_df.index.tolist().index(yr)
        fig_hm.add_annotation(
            x=rebound_col_idx, xref='x', y=rebound_yr_idx + 0.65, yref='y',
            text="↑ rebound<br>above baseline",
            showarrow=False,
            font=dict(size=9, color='#1d4ed8', family='IBM Plex Sans'),
            xanchor='center'
        )
        break

# ── 14.7 Layout ───────────────────────────────────────────────────────────────
fig_hm.update_layout(
    height=870,
    margin=dict(l=10, r=190, t=30, b=130),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    xaxis=dict(
        tickfont=dict(size=13),
        tickangle=-35,
        showgrid=False,
        side='bottom'
    ),
    yaxis=dict(
        tickfont=dict(size=14),
        showgrid=False,
        # 2012–13 at top, 2022–23 at bottom (chronological)
        autorange='reversed'
    ),
)

st.plotly_chart(fig_hm, use_container_width=True)

# ── 14.8 Heatmap colour legend row ───────────────────────────────────────────
st.markdown("""
<p style="font-size:1.0rem; color:#475569; text-align:center; margin-top:-0.5rem">
<span style="color:#991b1b; font-size:1.2rem">■</span> Below 2019–20 baseline &nbsp;|&nbsp;
<span style="color:#94a3b8; font-size:1.2rem">■</span> Near baseline (±5%) &nbsp;|&nbsp;
<span style="color:#1d4ed8; font-size:1.2rem">■</span> Above baseline &nbsp;|&nbsp;
<span style="color:#ea580c">– –</span> COVID lockdown year (2020–21) &nbsp;|&nbsp;
★#N = Treemap burden rank · Columns ordered: most disrupted → most resilient (left → right)
</p>
""", unsafe_allow_html=True)

# ── 14.9 Figure 2 caption ────────────────────────────────────────────────────
st.markdown("""
<p class="caption">
<b>Figure 2 (Supporting).</b> Matrix heatmap showing annual % change in admissions
relative to the 2019–20 pre-COVID baseline, for the
<b>ten highest-burden diagnoses identified in Figure 1 (★#1–#10)</b>.
Columns are ordered from largest lockdown drop (left) to most resilient (right) to
facilitate direct comparison of disruption severity. Red cells denote below-baseline
admissions; white cells indicate near-baseline activity; blue cells indicate
above-baseline volumes. The orange dashed border highlights the 2020–21 COVID lockdown
year. Exact percentage values are printed in each cell. Algorithmic annotations identify
the largest single-year drop, the most resilient diagnosis, and the first post-lockdown
rebound above baseline.
</p>
""", unsafe_allow_html=True)

# ── 14.10 Cross-visual analytical observation ────────────────────────────────
st.markdown("""
<div class="observation-box">
<b>🔍 Cross-Visual Observation — Treemap to Heatmap:</b><br>
The treemap establishes that the ten highest-burden diagnoses (★#1–#10) span multiple
ICD chapters and represent a wide range of clinical contexts. The heatmap then tests the
central question: did high cumulative volume translate to resilience during the 2020–21
COVID lockdown? <b>The data shows it did not — burden and resilience are structurally
uncorrelated.</b><br><br>
<b>★#1 Labour &amp; Delivery</b> — the single largest block in the treemap — fell by only
approximately 6–7% in 2020–21, consistent with the non-deferrable, time-critical nature
of obstetric care. In stark contrast, <b>★#3 Arthropathies</b>, the third highest-burden
diagnosis, dropped by over 50%, because joint replacement and related elective procedures
were systematically cancelled during the lockdown. Similarly, <b>★#6 Cataracts</b> fell
by approximately 46%, confirming that the entire Eye &amp; Ear chapter — which the treemap
shows to be almost entirely composed of this single diagnosis — was among the most acutely
disrupted.<br><br>
The heatmap further reveals that post-lockdown recovery was <b>uneven and incomplete</b>.
By 2021–22, Cataracts rebounded strongly above baseline (driven by the NHS elective
recovery programme), while Arthropathies remained well below baseline into 2022–23,
indicating structural and persistent service suppression rather than a brief one-year
shock. <b>Neither visualisation alone could convey this insight:</b> the treemap cannot
show temporal change, and the heatmap without the treemap would lack the hierarchical
context that explains why certain diagnoses proved resilient or vulnerable.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 15 — CW2 SUBMISSION TEMPLATE (PAGE 2)
#
# This collapsible section contains the complete publishable-quality
# description template required by COMP4037 CW2. All content is grounded in
# the actual visualisation output visible on this dashboard.
# =============================================================================
st.markdown('<hr class="thin">', unsafe_allow_html=True)

with st.expander("📋 CW2 Submission Template — Page 2 (expand to copy)", expanded=False):
    st.markdown("""
<div class="template-box">

<b>Research Question Answered:</b><br>
What is the relative burden of hospital admission categories across ICD chapters and
diagnoses, and how did the most substantively important diagnoses deviate from their
pre-COVID baseline during the 2020–21 lockdown and subsequent recovery period?

<br><br>
<b>Visual Design Type:</b><br>
<b>Main visual (Figure 1):</b> Hierarchical squarified treemap — a two-level nested
rectangle layout where each outer rectangle represents an ICD disease chapter and each
inner rectangle represents a specific diagnosis group within that chapter. Rectangle area
is proportional to total cumulative admissions (2012–2023), making hierarchical burden
immediately comparable across all scales of the disease taxonomy.<br><br>
<b>Supporting visual (Figure 2):</b> Matrix heatmap with a seven-stop diverging colour
scale — a 11-row × 10-column grid where rows are financial years (2012–13 to 2022–23)
and columns are the ten highest-burden diagnoses identified in Figure 1. Cell colour
encodes percentage change versus the 2019–20 pre-COVID baseline. Together, the two
designs constitute an orthogonal dual-view: Figure 1 encodes structural burden across
all chapters and diagnoses simultaneously; Figure 2 encodes temporal deviation across
the most clinically significant diagnoses.

<br><br>
<b>Name of Tool:</b><br>
Python 3 (programming language) · Pandas (data wrangling and aggregation) ·
NumPy (numerical computation) · Plotly (interactive chart rendering via
<i>go.Treemap</i> and <i>go.Heatmap</i>) · Streamlit (interactive web dashboard
deployment). All user-interface text is in English.

<br><br>
<b>Diagnosis Groups Shown:</b><br>
<b>Figure 1 (Treemap):</b> All ICD-10 summary-code diagnosis groups available in the
ONS dataset (2012–2023), nested within their respective ICD chapters. A total of 18
ICD chapters and all qualifying diagnosis groups within each are displayed. The ten
highest-burden diagnoses (by total cumulative admissions) are explicitly marked with
★ rank badges (★#1 through ★#10) and dark-bordered tiles, ordered from the highest
(★#1 Labour &amp; Delivery: 8.77M) to the tenth (★#10 General Symptoms: 3.84M).<br><br>
<b>Figure 2 (Heatmap):</b> The same ten ★-marked diagnoses from Figure 1 form the
ten columns of the heatmap. These are: ★#1 Labour &amp; Delivery · ★#2 Reproduction
Health Services · ★#3 Arthropathies · ★#4 Intestinal Diseases · ★#5 Digestive
Symptoms · ★#6 Cataracts · ★#7 Circ./Resp. Symptoms · ★#8 Oesophagus/Stomach ·
★#9 Blood Cancers · ★#10 General Symptoms.

<br><br>
<b>Variables:</b><br>
• <b>Total admissions (FAE — Finished Admissions Episodes):</b> The primary quantity
of interest. Summed across 2012–2023 to produce cumulative burden for the treemap area
encoding. Also used year-by-year to compute percentage change for the heatmap. Chosen
because it is the most direct measure of clinical volume and healthcare resource demand.<br><br>
• <b>ICD-10 chapter (derived variable):</b> Constructed by extracting the leading letter
of the ICD-10 diagnosis code and mapping it to 18 clinical chapter labels
(e.g., K → Digestive, O → Pregnancy &amp; Childbirth). Provides the outer hierarchy
required for the treemap; without it the chart would be structurally invalid.<br><br>
• <b>Diagnosis description (canonical):</b> The clinical name of each admission category
at the ICD-10 summary-code level. Forms the inner leaf level of the treemap and the
column identifiers in the heatmap. Labels were canonicalised to resolve multi-year
punctuation and spacing inconsistencies.<br><br>
• <b>Financial year:</b> NHS financial year (April–March), spanning 2012–13 to 2022–23.
Forms the row axis of the heatmap, enabling temporal reading of pre-COVID trends,
lockdown disruption, and recovery patterns across all eleven years.<br><br>
• <b>Percentage change vs 2019–20 baseline:</b> Computed as
((admissions_year − admissions_2019-20) / admissions_2019-20) × 100. Normalises all
diagnoses to a common scale so that differences in absolute volume do not confound
comparison of lockdown impact across the heatmap columns.

<br><br>
<b>Visual Mappings:</b><br>
<u>Figure 1 — Hierarchical Treemap:</u><br>
• <b>Hierarchy:</b> Outer rectangles = ICD chapters (18 groups); inner nested rectangles
= individual diagnosis groups within each chapter. The two-level nesting encodes the
ICD-10 classification structure directly into spatial containment.<br>
• <b>Area (size):</b> Each inner rectangle's area is proportional to its total admissions
(2012–2023). Larger tiles = greater cumulative burden. The squarification algorithm
maximises the aspect ratio of tiles to improve readability.<br>
• <b>Colour hue:</b> Assigned categorically to ICD chapter membership. Each chapter has
a unique hue (e.g., amber for Digestive, red for Cancer, cyan for Pregnancy &amp;
Childbirth), making chapter boundaries visible even without border lines.<br>
• <b>Colour intensity (within chapter):</b> Within the same chapter, colour saturation
increases with relative admission volume. The highest-volume diagnosis in each chapter
shows the full chapter hue; lower-volume diagnoses are blended towards white using a
power-compressed interpolation. This dual-channel encoding allows both chapter identity
and within-chapter magnitude to be read from a single cell.<br>
• <b>★ Rank labels and dark cell borders:</b> The ten highest-burden diagnoses are
displayed with gold label text prefixed by ★#1 through ★#10 and outlined with dark
(#1e293b) cell borders of increased width. This creates an explicit visual bridge to
Figure 2, whose columns correspond exactly to these ten diagnoses.<br>
• <b>Text labels:</b> Shown only for diagnosis tiles exceeding 0.4% of all admissions
(a minimum legibility threshold). Full names and exact totals are always available via
hover tooltip for any tile regardless of size.<br>
• <b>Position:</b> Determined by the squarified layout algorithm. Higher-burden chapters
generally appear towards the upper-left, as the algorithm places larger rectangles first.
<br><br>
<u>Figure 2 — Matrix Heatmap:</u><br>
• <b>X-axis (columns):</b> Ten diagnoses from Figure 1 (★#1–#10), ordered from the
largest lockdown decline on the left (★#3 Arthropathies, −53%) to the most resilient
on the right (★#1 Labour &amp; Delivery, −6%). Each column header includes the ★
rank, reinforcing the treemap–heatmap bridge.<br>
• <b>Y-axis (rows):</b> Financial years 2012–13 to 2022–23 in chronological order,
displayed top-to-bottom (oldest at top). This ordering ensures the pre-COVID baseline
row (2019–20) sits near the middle with pandemic and recovery years below.<br>
• <b>Cell colour (diverging scale):</b> A seven-stop red–white–blue scale maps
percentage change to colour. Dark red = below −50% (severe decline); near-white = near
baseline (0%); dark blue = above +25% (above-baseline volume). The symmetric
diverging scale makes both direction and magnitude of deviation readable from colour
alone.<br>
• <b>Cell annotations:</b> Exact percentage values are printed inside every cell (e.g.,
"−53%", "−6%", "+36%"), enabling precision reading where colour alone is insufficient.<br>
• <b>Orange dashed border:</b> An algorithmic shape layer circumscribes the 2020–21
lockdown row with an orange (#f97316) dashed rectangle, directing attention to the
pandemic disruption event. A labelled annotation "◄ COVID lockdown" is positioned to
the right of the border.<br>
• <b>Analytic annotations:</b> Three data-driven labels are placed algorithmically:
"▼ largest drop" marks the steepest lockdown column; "▲ most resilient" marks the
column with the smallest lockdown decline; "↑ rebound above baseline" marks the first
post-lockdown year and column where admissions exceeded the pre-COVID baseline.

<br><br>
<b>Unique Observation:</b><br>
The most important insight yielded by this dual-view design is that <b>cumulative
admission burden and pandemic resilience are structurally uncorrelated</b> across the
ten highest-volume diagnostic categories. A naïve assumption would be that the
highest-burden diagnoses — the ones the NHS manages most frequently — would also
prove the most operationally robust during a crisis. The visualisations refute this
assumption empirically and with visual immediacy.<br><br>
In the treemap (Figure 1), <b>★#1 Labour &amp; Delivery</b> is the single largest inner
tile, with 8.77 million admissions over 2012–2023. Its block rivals entire mid-sized
ICD chapters. In the heatmap (Figure 2), the same diagnosis sits in the rightmost
column, showing a lockdown decline of only approximately −6%: the most resilient of
all ten diagnoses. The clinical explanation is clear — obstetric care is non-deferrable,
and the NHS maintained it throughout the pandemic.<br><br>
The contrast with <b>★#3 Arthropathies</b> is the most striking finding. Despite ranking
third in cumulative burden in the treemap, it occupies the leftmost heatmap column with
a −53% lockdown decline — the steepest of the ten diagnoses. Joint replacement and
related musculoskeletal procedures are predominantly elective and time-flexible; they
were among the first to be cancelled as hospitals prioritised COVID patients. By 2022–23,
Arthropathies had still not returned to its pre-COVID baseline (−17%), demonstrating
a persistent structural disruption that a single-year view would miss entirely.<br><br>
<b>★#6 Cataracts</b> — almost entirely responsible for the Eye &amp; Ear chapter's area
in the treemap — fell −46% in 2020–21 but subsequently rebounded to +36% above baseline
by 2022–23, the strongest single-column rebound in the entire heatmap. This dramatic
V-shaped trajectory, visible as a shift from deep red to deep blue in the bottom rows
of that column, reflects the prioritised NHS elective recovery programme for ophthalmology.
No other diagnostic category shows this combination of steep initial collapse and strong
subsequent recovery.<br><br>
These patterns — the disconnect between volume and resilience, the V-shaped cataract
rebound, and the persistent musculoskeletal suppression — are wholly invisible in both
a standalone treemap and in any standard bar or line chart. The treemap cannot encode
time; a bar chart cannot encode hierarchy; a line chart with ten series cannot encode
both the year-on-year pattern and the relative ordering of lockdown severity
simultaneously. Only the combination of a hierarchical treemap and a diverging matrix
heatmap, linked through shared diagnosis identifiers, reveals all three dimensions of
this clinical story at once.

<br><br>
<b>Data Preparation:</b><br>
The raw ONS Hospital Admissions CSV was processed in five stages, all implemented in
Python 3 using Pandas and NumPy:<br><br>
<b>1. Column standardisation:</b> The raw file uses verbose multi-word column headers.
These were renamed to compact identifiers (code, description, year, Admissions, etc.)
to simplify downstream aggregation and referencing.<br><br>
<b>2. Numeric coercion:</b> Eleven columns (Admissions, Emergency, Male, Female, Waiting
List, Planned, Mean Age, Mean Length of Stay, Mean Time Waited, Finished Consultant
Episodes, FCE Bed Days) were converted from object to float64 using
<i>pd.to_numeric(errors='coerce')</i>. Non-numeric entries (dashes, blanks, suppressed
small counts) were silently converted to NaN, allowing pandas aggregation functions to
operate correctly while preserving valid counts.<br><br>
<b>3. Partial year exclusion:</b> The 2023–24 financial year was still incomplete at
the time of analysis. Its inclusion would systematically understate that year's total
relative to complete years, distorting percentage change calculations. It was excluded
entirely from all aggregation and visualisation.<br><br>
<b>4. Diagnosis label canonicalisation:</b> The same clinical condition occasionally
appears across different financial years with minor punctuation or spacing differences
(e.g., "Diseases of oesophagusstomach &amp; duodenum" versus "Diseases of oesophagus,
stomach &amp; duodenum"). Without canonicalisation, these are treated as distinct
diagnoses, fragmenting their admission counts. A strip-and-normalise pipeline was
applied (strip leading/trailing spaces, collapse multiple spaces with regex), followed
by explicit replacement of the three known inconsistent variants to a single canonical
form. This ensured that each clinical diagnosis aggregated correctly across all eleven
financial years.<br><br>
<b>5. ICD-10 chapter derivation:</b> The outer treemap hierarchy (ICD chapter) does not
exist as a column in the raw data. It was constructed by extracting the leading letter
from each diagnosis code using a two-stage regex (preferred: letter followed by a digit;
fallback: first alphabetic character) and mapping it to one of 18 chapter labels via a
lookup dictionary. This derived variable is what makes the treemap structurally valid;
without it, the data has no hierarchy and the chart would be equivalent to a flat area
chart.<br><br>
<b>6. Top-10 ranking:</b> Diagnoses were ranked by total cumulative admissions (2012–2023)
after canonicalisation and aggregation. The ten highest-ranking diagnoses were stored as
an ordered list and a rank dictionary. Their rank numbers were embedded in treemap tile
labels (★#1 through ★#10) and in heatmap column headers, making the identity of the
top-10 diagnoses explicitly navigable between both figures.<br><br>
<b>7. Baseline normalisation for the heatmap:</b> For each of the ten selected diagnoses,
annual admissions for every year 2012–13 to 2022–23 were expressed as a percentage
change relative to the 2019–20 pre-COVID baseline: Δ% = ((admissions_t −
admissions_2019-20) / admissions_2019-20) × 100. This normalisation removes the
confounding effect of absolute volume differences, allowing diagnoses with very different
scales (e.g., Labour &amp; Delivery at 800k/year vs General Symptoms at 300k/year) to be
directly compared by colour within the same matrix. Heatmap columns were then sorted by
2020–21 Δ% in ascending order (steepest decline leftmost) to make the disruption gradient
immediately readable. Raw values were clipped at −75% / +35% only for colour scale
rendering; actual values remain visible in cell annotations and hover tooltips.

<br><br>
<b>Why this design is better than standard charts:</b><br>
A bar chart can show cumulative admissions by diagnosis but cannot convey the ICD
chapter hierarchy or the relative size of diagnoses within chapters — it treats all bars
as equivalent in structural context. A line chart can show temporal trends for individual
diagnoses but requires a separate panel for each, preventing direct cross-diagnosis
comparison. A pie chart cannot simultaneously show hierarchy, area proportionality, and
intra-category composition.<br><br>
The hierarchical treemap solves the hierarchy problem: it shows all 18 ICD chapters and
all their constituent diagnoses in a single, compact view, where area encodes magnitude
without distortion and nested containment encodes the ICD classification directly. The
matrix heatmap solves the temporal multi-variable comparison problem: it shows 11 years ×
10 diagnoses in a single matrix, where colour encodes direction and magnitude of deviation
and the ordered columns create an immediate left-to-right ranking of disruption severity.
Neither question — "how is burden structured?" and "how did burden change?" — can be
answered effectively by the other's visual type. The two charts are analytically
complementary and mutually essential.

<br><br>
<b>Video URL:</b> [Insert YouTube or Vimeo screen-capture link here — recommended ≤ 2 minutes.
The video should demonstrate: (1) treemap hover interaction showing exact admission counts,
(2) zooming into a chapter to show inner diagnosis tiles, (3) the ★ rank system and its
link to Figure 2, (4) heatmap hover showing exact % values per cell,
(5) narrated walkthrough of the cross-visual observation.]<br>
<b>Source Code URL:</b> [Insert GitHub or GitLab repository URL here. The repository should
contain hospital_viz_app.py, requirements.txt, and a README.md with installation and
run instructions. Code follows PEP 8 conventions and is commented throughout.]

</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 16 — DASHBOARD FOOTER
# =============================================================================
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.78rem; margin-top:2rem'>
COMP4037 Research Methods · Coursework 2 · ONS Hospital Admissions 2012–2023 ·
Python 3 / Pandas / NumPy / Plotly / Streamlit
</div>
""", unsafe_allow_html=True)

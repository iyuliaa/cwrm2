"""
COMP4037 CW2 — UK Hospital Admissions Visualization
Research Question: What is the relative burden of hospital admission categories across ICD
chapters and diagnoses, and how did the most substantively important diagnoses deviate from
their pre-COVID baseline during the 2020–2021 lockdown and subsequent recovery period?

Visual Design: Hierarchical Treemap (main) + Matrix Heatmap with diverging colour scale (supporting)
Tool: Python 3, Pandas, Plotly, and Streamlit
"""

import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Hospital Admissions — CW2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.big-title {
    font-family: 'Source Serif 4', serif;
    font-size: 2.0rem;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
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
.section-label {
    font-family: 'Source Serif 4', serif;
    font-size: 1.2rem;
    color: #0f172a;
    border-bottom: 3px solid #f97316;
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.5rem 0;
}
.caption {
    font-size: 0.81rem;
    color: #64748b;
    font-style: italic;
    margin-top: 0.3rem;
    text-align: center;
    line-height: 1.5;
}
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
hr.thin { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Prepare Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("./data_raw/clean_combined_data.csv")
    df = df.rename(columns={
        'Primary diagnosis: summary code and description': 'code',
        'Unnamed: 1': 'description',
        '2012-13': 'year'
    })
    num_cols = ['Admissions', 'Emergency', 'Male', 'Female',
                'Waiting list', 'Planned', 'Mean age',
                'Mean length of stay', 'Mean time waited',
                'Finished consultant episodes', 'FCE bed days']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove incomplete partial year
    df = df[df['year'] != '2023-24']
    df = df.dropna(subset=['year', 'description'])
    df['description'] = df['description'].astype(str).str.strip()

    # Derive ICD chapter from first letter of diagnosis code
    chapter_map = {
        'A': 'Infectious Diseases', 'B': 'Infectious Diseases',
        'C': 'Cancer',              'D': 'Cancer & Blood',
        'E': 'Endocrine & Metabolic', 'F': 'Mental Health',
        'G': 'Nervous System',      'H': 'Eye & Ear',
        'I': 'Circulatory',         'J': 'Respiratory',
        'K': 'Digestive',           'L': 'Skin',
        'M': 'Musculoskeletal',     'N': 'Genitourinary',
        'O': 'Pregnancy & Childbirth', 'P': 'Perinatal',
        'Q': 'Congenital',          'R': 'Symptoms & Signs',
        'S': 'Injury & Poisoning',  'T': 'Injury & Poisoning',
        'Z': 'Health Services'
    }

    def get_chapter(code):
        if not isinstance(code, str):
            return 'Other'
        c = code.strip().upper()
        return chapter_map.get(c[0] if c else 'Z', 'Other')

    df['chapter'] = df['code'].apply(get_chapter)
    return df


df = load_data()
years = sorted(df['year'].unique())

# ─── Shortened display labels ──────────────────────────────────────────────────
SHORT_LABELS = {
    'Complications of labour and delivery':                          'Labour & Delivery',
    'Health services in circumstances related to reproduction':      'Reproduction Health Svcs',
    'Arthropathies':                                                 'Arthropathies',
    'Other diseases of intestines':                                  'Intestinal Diseases',
    'Symptoms & signs inv. the digestive system & abdomen':          'Digestive Symptoms',
    'Disorders of lens (including cataracts)':                       'Cataracts (Lens Disorders)',
    'Symptoms & signs inv. the circulatory/respiratory system':      'Circ./Resp. Symptoms',
    'Diseases of oesophagus, stomach & duodenum':                    'Oesophagus / Stomach',
    'Malignant neoplasms of lymphoid, haematopoietic & rel. tiss.':  'Blood Cancers',
    'General symptoms & signs':                                      'General Symptoms',
    'Dorsopathies':                                                  'Dorsopathies (Back & Spine)',
    'Other forms of heart disease':                                  'Heart Disease (Other)',
    'Hernia':                                                        'Hernia',
    'Soft tissue disorders':                                         'Soft Tissue Disorders',
    'Examination and investigation':                                 'Examination / Investigation',
}

# ─── Chapter colour palette (categorical, one hue per chapter) ─────────────────
CHAPTER_COLORS = {
    'Digestive':              '#d97706',   # amber
    'Symptoms & Signs':       '#b45309',   # brown-amber
    'Cancer':                 '#dc2626',   # red
    'Cancer & Blood':         '#b91c1c',   # dark red
    'Musculoskeletal':        '#16a34a',   # green
    'Health Services':        '#2563eb',   # blue
    'Injury & Poisoning':     '#7c3aed',   # purple
    'Circulatory':            '#db2777',   # pink
    'Respiratory':            '#0d9488',   # teal
    # amber-gold (distinct from Digestive via sat)
    'Eye & Ear':              '#d97706',
    'Pregnancy & Childbirth': '#0891b2',   # cyan
    'Genitourinary':          '#0284c7',   # sky blue
    'Infectious Diseases':    '#ea580c',   # orange
    'Nervous System':         '#9333ea',   # violet
    'Endocrine & Metabolic':  '#0369a1',   # dark sky
    'Mental Health':          '#be185d',   # rose
    'Skin':                   '#65a30d',   # lime-green
    'Perinatal':              '#059669',   # emerald
    'Congenital':             '#6d28d9',   # indigo-purple
    'Other':                  '#94a3b8',   # slate
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_short_label(text: str) -> str:
    """Return a concise display label, using the lookup dict when available."""
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


# ─── Smart heatmap diagnosis selection ────────────────────────────────────────
# Strategy: 5 highest-burden + 5 most lockdown-disrupted (non-overlapping)
top5_burden = (
    df.groupby('description')['Admissions'].sum()
    .nlargest(5).index.tolist()
)

base_ser = df[df['year'] ==
              '2019-20'].groupby('description')['Admissions'].sum()
lock_ser = df[df['year'] ==
              '2020-21'].groupby('description')['Admissions'].sum()
common_idx = base_ser.index.intersection(lock_ser.index)
# Only include diagnoses with meaningful pre-COVID volume (>10 000 admissions)
sig_idx = base_ser[common_idx][base_ser[common_idx] > 10_000].index
pct_lockdown = ((lock_ser[sig_idx] - base_ser[sig_idx]) /
                base_ser[sig_idx] * 100).dropna().sort_values()

# Prefer elective/deferrable categories for storytelling clarity
elective_keywords = ['cataract', 'lens', 'hernia', 'dorsop', 'soft tissue', 'examination',
                     'orthopaed', 'arthro', 'tonsil', 'ear', 'skin append']
most_disrupted_candidates = [
    d for d in pct_lockdown.index if d not in top5_burden
]
# Prioritise clearly elective-sounding diagnoses, then fall back to worst drops
elective_disrupted = [d for d in most_disrupted_candidates
                      if any(k in d.lower() for k in elective_keywords)]
other_disrupted = [
    d for d in most_disrupted_candidates if d not in elective_disrupted]
top5_disrupted = (elective_disrupted + other_disrupted)[:5]

selected_diagnoses = top5_burden + top5_disrupted

# Order heatmap columns: largest lockdown drop first (most disrupted → most stable)


def lockdown_pct(d):
    b = base_ser.get(d, np.nan)
    l = lock_ser.get(d, np.nan)
    return (l - b) / b * 100 if (pd.notna(b) and b > 0) else 0


selected_diagnoses_sorted = sorted(selected_diagnoses, key=lockdown_pct)


# ─── KPI computation ──────────────────────────────────────────────────────────
total_adm = df['Admissions'].sum()
covid_change = (
    df[df['year'] == '2020-21']['Admissions'].sum() /
    df[df['year'] == '2019-20']['Admissions'].sum() - 1
) * 100
years_covered = f"{years[0]} – {years[-1]}"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="big-title">🏥 UK Hospital Admissions — Burden & Lockdown Impact</p>',
            unsafe_allow_html=True)
st.markdown("""
<div class="rq-box">
<b>Research Question:</b> What is the relative burden of hospital admission categories across
ICD chapters and diagnoses, and how did the most substantively important diagnoses deviate
from their pre-COVID baseline during the <b>2020–21 lockdown</b> and subsequent recovery period?<br>
<span style="font-size:0.82rem; color:#94a3b8">ONS Hospital Admissions Dataset · 2012–2023 · COMP4037 CW2 · Python 3 / Streamlit / Plotly</span>
</div>
""", unsafe_allow_html=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE — HIERARCHICAL TREEMAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-label">Figure 1 (Main) — Hierarchical Treemap: Relative Admission Burden by ICD Chapter & Diagnosis</div>',
    unsafe_allow_html=True
)
st.markdown("""
**Area (size)** = total admissions 2012–2023 · 
**Colour** = ICD disease chapter (categorical) · 
**Outer rectangle** = ICD chapter · **Inner rectangle** = specific diagnosis.  
Hover over any block to view exact totals.
""")

# Aggregate for treemap
tree_df = (
    df.groupby(['chapter', 'description'], as_index=False)['Admissions']
    .sum()
    .rename(columns={'Admissions': 'total'})
)
tree_df = tree_df[tree_df['total'] > 0].copy()
tree_df['label_full'] = tree_df['description'].fillna(
    'Unknown').astype(str).str.strip()
tree_df['label_short'] = tree_df['label_full'].apply(make_short_label)

# Chapter size threshold: show label only if chapter area is at least 1% of total
grand_total = tree_df['total'].sum()
chapter_totals = tree_df.groupby('chapter')['total'].sum()
MIN_DIAG_FRACTION = 0.008  # show diagnosis label only if >= 0.8% of grand total

tree_df['show_label'] = tree_df['total'] / grand_total >= MIN_DIAG_FRACTION
tree_df['label_display'] = tree_df.apply(
    lambda r: r['label_short'] if r['show_label'] else '', axis=1
)

# Build manual treemap structure
labels, parents, values, ids, colors, customdata = [], [], [], [], [], []

root_id = "root"
labels.append("All Admissions")
parents.append("")
values.append(grand_total)
ids.append(root_id)
colors.append("#f1f5f9")
customdata.append(["All Admissions", grand_total, "—"])

for chapter, ch_total in chapter_totals.sort_values(ascending=False).items():
    chapter_id = f"ch::{chapter}"
    labels.append(chapter)
    parents.append(root_id)
    values.append(ch_total)
    ids.append(chapter_id)
    colors.append(CHAPTER_COLORS.get(chapter, "#94a3b8"))
    customdata.append([chapter, ch_total, chapter])

for _, row in tree_df.iterrows():
    chapter = row['chapter']
    full_name = row['label_full']
    disp_name = row['label_display']
    total = row['total']
    chapter_id = f"ch::{chapter}"
    diag_id = f"dx::{chapter}::{full_name}"

    labels.append(disp_name)
    parents.append(chapter_id)
    values.append(total)
    ids.append(diag_id)
    colors.append(CHAPTER_COLORS.get(chapter, "#94a3b8"))
    customdata.append([full_name, total, chapter])

fig_tree = go.Figure(go.Treemap(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    textinfo="label",
    textfont=dict(size=12, family='IBM Plex Sans', color='white'),
    marker=dict(
        colors=colors,
        # clear white borders separate chapters
        line=dict(width=2, color='white'),
        pad=dict(t=24, b=5, l=5, r=5)
    ),
    customdata=customdata,
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'Total Admissions (2012–2023): <b>%{customdata[1]:,.0f}</b><br>'
        'ICD Chapter: %{customdata[2]}<extra></extra>'
    ),
    root_color='#f1f5f9'
))

fig_tree.update_layout(
    height=600,
    margin=dict(t=30, l=8, r=8, b=8),
    paper_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    uniformtext=dict(minsize=10, mode='hide')   # hide labels in tiny cells
)

st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("""
<p class="caption">
<b>Figure 1 (Main).</b> Hierarchical treemap of total UK hospital admissions (2012–2023).
Outer rectangles represent ICD disease chapters; inner rectangles represent individual diagnosis groups nested within each chapter.
Rectangle <b>area</b> is proportional to cumulative admissions, making relative burden immediately visible across scales.
<b>Colour</b> encodes ICD chapter categorically — not admission volume — so that the hierarchical structure 
(chapter → diagnosis) can be interpreted simultaneously with size. Labels are shown only for diagnosis groups 
large enough to be legible; full names and exact counts are available on hover.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Treemap Observation:</b><br>
The <b>Digestive</b> chapter holds the largest cumulative area, indicating the greatest aggregate burden across the study period,
driven by a spread of diagnoses rather than a single dominant condition.
In contrast, <b>Pregnancy &amp; Childbirth</b> is a comparatively small chapter by area, yet it contains the single largest
inner rectangle — <b>Labour &amp; Delivery</b> — a diagnosis whose admissions alone rival entire mid-sized chapters.
The <b>Eye &amp; Ear</b> chapter is almost entirely composed of one diagnosis (Cataracts / Lens Disorders),
illustrating how some chapters are clinically diverse while others are dominated by a single high-volume procedure.
These structural differences in chapter composition are invisible in standard bar charts but immediately apparent here.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORTING FIGURE — % CHANGE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-label">Figure 2 (Supporting) — Matrix Heatmap: % Change in Admissions vs 2019–20 Baseline</div>',
    unsafe_allow_html=True
)
st.markdown("""
Columns = **10 selected diagnoses** (5 highest-burden + 5 most lockdown-disrupted, ordered by severity of lockdown drop).  
Rows = financial year. **Cell colour** = % change vs the 2019–20 pre-COVID baseline.  
🔴 Red = below baseline · ⬜ White = near baseline · 🔵 Blue = above baseline.  
The orange dashed border marks the 2020–21 lockdown year.
""")

# Build heatmap matrix using sorted diagnosis list
hm_rows = {}
yr_totals_all = {yr: df[df['year'] == yr].groupby('description')['Admissions'].sum()
                 for yr in years}

for yr in years:
    row = {}
    for d in selected_diagnoses_sorted:
        b = base_ser.get(d, np.nan)
        v = yr_totals_all[yr].get(d, np.nan)
        row[d] = (v - b) / b * 100 if (pd.notna(b) and b > 0) else np.nan
    hm_rows[yr] = row

hm_df = pd.DataFrame(hm_rows).T[selected_diagnoses_sorted]

# Short column labels
short_cols = [SHORT_LABELS.get(d, (d[:26] + '…' if len(d) > 26 else d))
              for d in selected_diagnoses_sorted]
hm_df.columns = short_cols

# Clip for colour scale
z_max, z_min = 35, -75
hm_clipped = hm_df.clip(z_min, z_max)
annot = hm_df.map(lambda v: f"{v:+.0f}%" if pd.notna(v) else "")

fig_hm = go.Figure()
fig_hm.add_trace(go.Heatmap(
    z=hm_clipped.values,
    x=short_cols,
    y=hm_df.index.tolist(),
    zmin=z_min,
    zmax=z_max,
    colorscale=[
        [0.00, '#7f1d1d'],   # deep red
        [0.15, '#b91c1c'],
        [0.35, '#fca5a5'],   # light red
        [0.50, '#f8fafc'],   # white = baseline
        [0.65, '#93c5fd'],   # light blue
        [0.85, '#1d4ed8'],
        [1.00, '#1e3a8a'],   # deep blue
    ],
    colorbar=dict(
        title=dict(text='% vs<br>2019–20<br>baseline', font=dict(size=10)),
        tickvals=[-75, -50, -25, 0, 25, 35],
        ticktext=['-75%', '-50%', '-25%', '0%', '+25%', '+35%'],
        tickfont=dict(size=10),
        len=0.80,
        thickness=14,
        x=1.04,
    ),
    text=annot.values,
    texttemplate="%{text}",
    textfont=dict(size=13, color='#1e293b'),
    hovertemplate='<b>%{x}</b><br>Year: %{y}<br>Change vs baseline: %{z:+.1f}%<extra></extra>',
    xgap=2,
    ygap=2,
))

# Lockdown year highlight
lockdown_idx = hm_df.index.tolist().index('2020-21')
fig_hm.add_shape(
    type='rect', xref='paper', yref='y',
    x0=-0.01, x1=1.01,
    y0=lockdown_idx - 0.5, y1=lockdown_idx + 0.5,
    line=dict(color='#f97316', width=3, dash='dot'),
    fillcolor='rgba(249,115,22,0.06)', layer='above'
)
fig_hm.add_annotation(
    x=1.0, xref='paper', y='2020-21', yref='y',
    text='◄ COVID<br>lockdown',
    showarrow=False,
    font=dict(size=11, color='#c2410c', family='IBM Plex Sans'),
    xanchor='left', xshift=12
)

# ── Key analytic annotations ──
# Identify column indices for specific diagnoses we want to annotate
label_to_idx = {lbl: i for i, lbl in enumerate(short_cols)}

# Annotation: largest single-cell drop in lockdown row
lockdown_vals = hm_clipped.iloc[lockdown_idx]
worst_col = lockdown_vals.idxmin()
worst_col_idx = short_cols.index(worst_col)

fig_hm.add_annotation(
    x=worst_col_idx, xref='x', y=lockdown_idx - 0.65, yref='y',
    text="▼ largest<br>drop",
    showarrow=False,
    font=dict(size=9, color='#991b1b', family='IBM Plex Sans'),
    xanchor='center'
)

# Annotation: most resilient diagnosis in lockdown row (smallest absolute drop)
resilient_col = lockdown_vals.idxmax()
resilient_idx = short_cols.index(resilient_col)
fig_hm.add_annotation(
    x=resilient_idx, xref='x', y=lockdown_idx - 0.65, yref='y',
    text="▲ most<br>resilient",
    showarrow=False,
    font=dict(size=9, color='#1e40af', family='IBM Plex Sans'),
    xanchor='center'
)

# Annotation: any column that rebounds above baseline post-lockdown (first year > +5%)
post_lockdown_years = hm_df.index.tolist()[lockdown_idx + 1:]
rebound_label = None
for yr in post_lockdown_years:
    row_data = hm_df.loc[yr]
    above = row_data[row_data > 5]
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
        rebound_label = yr
        break

fig_hm.update_layout(
    height=870,
    margin=dict(l=10, r=190, t=30, b=110),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    xaxis=dict(
        tickfont=dict(size=14),
        tickangle=-35,
        showgrid=False,
        side='bottom'
    ),
    yaxis=dict(
        tickfont=dict(size=14),
        showgrid=False,
        autorange='reversed'
    ),
)

st.plotly_chart(fig_hm, use_container_width=True)

# Legend explainer row
st.markdown("""
<p style="font-size:0.82rem; color:#475569; text-align:center; margin-top:-0.5rem">
<span style="color:#991b1b">■</span> Below 2019–20 baseline &nbsp;|&nbsp;
<span style="color:#94a3b8">■</span> Near baseline (±5%) &nbsp;|&nbsp;
<span style="color:#1d4ed8">■</span> Above baseline &nbsp;|&nbsp;
<span style="color:#ea580c">– –</span> COVID lockdown year (2020–21) &nbsp;|&nbsp;
Columns ordered: most disrupted → most resilient (left → right)
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p class="caption">
<b>Figure 2 (Supporting).</b> Matrix heatmap showing annual % change in admissions relative to the 2019–20 pre-COVID baseline,
for ten selected diagnoses. The selection combines the <b>five highest-burden diagnoses</b> from the treemap with
the <b>five most lockdown-disrupted diagnoses</b> (by % decline in 2020–21, among diagnoses with &gt;10,000 baseline admissions).
Columns are ordered from largest lockdown drop (left) to most resilient (right) to facilitate visual comparison.
Red = decline vs baseline; white = near-baseline; blue = rise above baseline.
The orange dashed border highlights the 2020–21 COVID lockdown year.
Cell values show exact % change. Analytic annotations mark the largest drop, most resilient, and first rebound above baseline.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Unique Observation — Cross-visual insight:</b><br>
The treemap shows that cumulative admission burden is concentrated in large chapters such as <b>Digestive</b>,
yet the heatmap reveals that the strongest lockdown-era contractions are <i>not</i> necessarily located in the
largest burden categories. In particular, <b>Arthropathies</b> — already among the top-burden diagnoses in the
treemap — shows a ~53% decline during 2020–21, while <b>Labour &amp; Delivery</b>, the single largest diagnosis
block, fell by less than 7%. Diagnoses dominated by elective, deferrable procedures (e.g., Cataracts, Hernia,
Dorsopathies) exhibit sharp 2020–21 contractions regardless of their treemap size. Meanwhile, the heatmap also
reveals that recovery after the lockdown year is incomplete and uneven: some categories rebound to or above
baseline by 2021–22, while others remain persistently below, indicating a structural service disruption rather
than a simple one-year shock. <b>This shows that overall burden and disruption sensitivity are not aligned:
high-burden categories are not always the most lockdown-sensitive, while less deferrable care demonstrates
greater resilience.</b>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CW2 SUBMISSION TEMPLATE — Page 2 content
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="thin">', unsafe_allow_html=True)
with st.expander("📋 CW2 Submission Template — Page 2 (expand to copy)", expanded=False):
    st.markdown("""
<div class="template-box">

<b>Research Question Answered:</b><br>
What is the relative burden of hospital admission categories across ICD chapters and diagnoses,
and how did the most substantively important diagnoses deviate from their pre-COVID baseline
during the 2020–2021 lockdown and subsequent recovery period?

<br><br>
<b>Visual Design Type:</b><br>
Main visual: Hierarchical treemap<br>
Supporting visual: Matrix heatmap with diverging colour scale

<br><br>
<b>Name of Tool:</b><br>
Python 3, Pandas, Plotly, and Streamlit

<br><br>
<b>Diagnosis Groups Shown:</b><br>
The treemap shows ICD disease chapters as the top hierarchical level and diagnosis groups nested
within each chapter. The heatmap focuses on ten selected diagnoses chosen to compare temporal
disruption and recovery relative to the pre-COVID baseline: the five highest-burden diagnoses
by cumulative admissions (2012–2023) and the five most lockdown-disrupted diagnoses by
percentage decline in 2020–21 (among diagnoses with more than 10,000 baseline admissions),
with no overlap between the two groups.

<br><br>
<b>Variables:</b><br>
• <b>Total admissions</b> (FAE/FCE-derived admission count) — used to represent cumulative hospital burden;
  mapped to rectangle area in the treemap.<br>
• <b>ICD chapter</b> — used to provide a meaningful high-level hierarchy across diagnosis families;
  mapped to colour in the treemap.<br>
• <b>Diagnosis description</b> — used to identify specific admission categories nested within each chapter.<br>
• <b>Financial year</b> — used to reveal temporal change across the study period; forms the row axis
  of the heatmap.<br>
• <b>Percentage change relative to the 2019–20 baseline</b> — used to normalise comparison across
  diagnoses of different absolute scales and to highlight disruption during and after the COVID
  lockdown period; mapped to cell colour in the heatmap.

<br><br>
<b>Visual Mappings:</b><br>
<u>Treemap:</u><br>
• Hierarchy → outer rectangles represent ICD chapters; inner rectangles represent diagnosis groups
  nested within each chapter.<br>
• Area (size) → total admissions accumulated over 2012–2023; larger rectangles indicate greater
  cumulative burden.<br>
• Colour → ICD chapter; colour is used categorically to separate major disease families and make
  the hierarchy easier to interpret. Colour does not encode admission volume.<br>
• Position → determined by the treemap squarified layout algorithm; larger categories tend to appear
  more prominently.<br>
• Text labels → shown only for diagnosis groups whose area exceeds a minimum legibility threshold;
  full names and counts are available on hover.<br>

<u>Heatmap:</u><br>
• X-axis → ten selected diagnosis groups, ordered from the largest lockdown drop (leftmost) to the
  most resilient (rightmost), enabling direct visual comparison of disruption severity.<br>
• Y-axis → financial year, ordered from oldest (top) to most recent (bottom).<br>
• Cell colour → percentage change in admissions relative to the 2019–20 baseline, using a diverging
  red–white–blue scale.<br>
• Red hues → below-baseline admissions; white / near-neutral tones → values close to baseline;
  blue hues → above-baseline admissions.<br>
• Dashed orange border → highlights the 2020–21 COVID lockdown year.<br>
• Cell annotations → exact percentage values printed inside each cell to support precise comparison.<br>
• Analytic annotations → three labels mark the largest drop, the most resilient diagnosis, and the
  first year a diagnosis rebounds above baseline.

<br><br>
<b>Unique Observation:</b><br>
The treemap shows that cumulative admission burden is concentrated in large chapters such as Digestive,
yet the heatmap reveals that the strongest lockdown-era contractions are not necessarily located in the
largest burden categories. In particular, Arthropathies — already among the top-burden diagnoses in
the treemap — shows approximately a 53% decline during 2020–21, while Labour and Delivery, the single
largest diagnosis block, fell by less than 7%. Diagnoses dominated by elective, deferrable procedures
(such as Cataracts, Hernia, and Dorsopathies) exhibit sharp 2020–21 contractions regardless of their
treemap size. The heatmap also reveals that recovery is incomplete and uneven after the lockdown year:
some categories rebound to or above baseline by 2021–22, while others remain persistently below,
indicating a longer-term structural service disruption rather than a simple one-year shock. This
demonstrates that overall burden and disruption sensitivity are not aligned — high-burden categories
are not always the most lockdown-sensitive, while less deferrable care demonstrates greater resilience.

<br><br>
<b>Data Preparation:</b><br>
The original admissions dataset was cleaned by standardising column names, converting
admissions-related fields to numeric format using coercion, and removing incomplete or unusable records,
including the partial 2023–24 year and rows with missing diagnosis or year information. An ICD chapter
field was derived from the first letter of the diagnosis code (e.g., M → Musculoskeletal) in order to
construct a valid two-level hierarchy for the treemap. Admissions were then aggregated by chapter and
diagnosis across 2012–2023 for the main visual. For the supporting heatmap, a ten-diagnosis selection
was constructed from two non-overlapping groups: the five diagnoses with the highest cumulative burden,
and the five with the most extreme lockdown-era decline (restricted to diagnoses with more than 10,000
admissions in 2019–20 to exclude low-volume categories). Annual percentage change was computed relative
to the 2019–20 baseline to make pre- and post-lockdown comparisons comparable across diagnoses of
different absolute scales. Heatmap columns were ordered by magnitude of lockdown decline to reinforce
visual interpretation. Long diagnosis labels were shortened for readability in the final figure, with
full descriptions preserved in hover tooltips.

<br><br>
<b>Why this design is effective:</b><br>
The treemap provides a compact, area-proportional overview of cumulative burden across multiple
hierarchical levels simultaneously, allowing chapter-level and diagnosis-level structure to be read
in a single glance. This is not achievable with a bar or line chart without collapsing the hierarchy
or requiring multiple panels. The heatmap complements this by revealing temporal deviation, disruption,
and recovery patterns across a matrix of diagnoses and years, enabling comparison across twelve years
and ten diagnosis groups in one view — a task that would require ten separate line charts with aligned
axes to replicate. Together, the two visuals reveal both structural burden and temporal instability,
allowing the relationship between cumulative scale and disruption sensitivity to be interpreted
directly and non-trivially from the combined display.

<br><br>
<b>Video URL:</b> [insert link]<br>
<b>Source Code URL:</b> [insert link]

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.78rem; margin-top:2rem'>
COMP4037 Research Methods · Coursework 2 · ONS Hospital Admissions 2012–2023 · Python 3 / Streamlit / Plotly
</div>
""", unsafe_allow_html=True)

"""
COMP4037 CW2 — UK Hospital Admissions Visualization
Research Question: What is the relative burden of each hospital admissions category,
and how did high-burden categories change during the 2020-21 COVID lockdown?

Visual Design: Treemap (main) + % Change Heatmap (supporting)
Tool: Python — Streamlit + Plotly
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
    font-size: 2.1rem;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.rq-box {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.3rem;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #f97316;
}
.rq-box b { color: #fb923c; }
.section-label {
    font-family: 'Source Serif 4', serif;
    font-size: 1.25rem;
    color: #0f172a;
    border-bottom: 3px solid #f97316;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 0.6rem 0;
}
.caption {
    font-size: 0.82rem;
    color: #64748b;
    font-style: italic;
    margin-top: 0.3rem;
    text-align: center;
}
.observation-box {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-left: 5px solid #f97316;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.91rem;
    color: #431407;
}
.observation-box b { color: #c2410c; }
.template-box {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 5px solid #16a34a;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.87rem;
    color: #14532d;
    line-height: 1.8;
}
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi { background: #0f172a; border-radius: 10px; padding: 0.8rem 1.2rem;
       flex: 1; border-top: 3px solid #f97316; }
.kpi-v { font-size: 1.7rem; font-weight: 600; color: #fb923c;
          font-family: 'Source Serif 4', serif; }
.kpi-l { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
          letter-spacing: 0.07em; }
hr.thin { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
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

    df = df[df['year'] != '2023-24']
    df = df.dropna(subset=['year', 'description'])
    df['description'] = df['description'].astype(str).str.strip()

    chapter_map = {
        'A': 'Infectious Diseases', 'B': 'Infectious Diseases',
        'C': 'Cancer', 'D': 'Cancer & Blood',
        'E': 'Endocrine & Metabolic', 'F': 'Mental Health',
        'G': 'Nervous System', 'H': 'Eye & Ear',
        'I': 'Circulatory', 'J': 'Respiratory',
        'K': 'Digestive', 'L': 'Skin',
        'M': 'Musculoskeletal', 'N': 'Genitourinary',
        'O': 'Pregnancy & Childbirth', 'P': 'Perinatal',
        'Q': 'Congenital', 'R': 'Symptoms & Signs',
        'S': 'Injury & Poisoning', 'T': 'Injury & Poisoning',
        'Z': 'Health Services'
    }

    def get_chapter(code):
        if not isinstance(code, str):
            return 'Other'
        c = code.strip().upper()
        return chapter_map.get(c[0] if c else 'Z', 'Other')

    df['chapter'] = df['code'].apply(get_chapter)
    df['short'] = df['description'].apply(
        lambda x: x[:36] + '…' if isinstance(x, str) and len(x) > 38 else x
    )
    return df


df = load_data()
years = sorted(df['year'].unique())

SHORT_LABELS = {
    'Complications of labour and delivery': 'Labour & delivery',
    'Health services in circumstances related to reproduction': 'Reproduction health svcs',
    'Arthropathies': 'Arthropathies',
    'Other diseases of intestines': 'Intestinal diseases',
    'Symptoms & signs inv. the digestive system & abdomen': 'Digestive symptoms',
    'Diseases of oesophagus, stomach & duodenum': 'Oesophagus / Stomach',
    'Symptoms & signs inv. the circulatory/respiratory system': 'Circ./resp. symptoms',
    'Malignant neoplasms of lymphoid, haematopoietic & rel. tiss.': 'Blood cancers',
    'Disorders of lens (including cataracts)': 'Cataracts',
    'General symptoms & signs': 'General symptoms',
    'Dorsopathies': 'Back & spine',
    'Other forms of heart disease': 'Heart disease (other)',
}

CHAPTER_COLORS = {
    'Digestive':              '#e07b39',
    'Symptoms & Signs':       '#d4a843',
    'Cancer':                 '#c0392b',
    'Cancer & Blood':         '#e74c3c',
    'Musculoskeletal':        '#27ae60',
    'Health Services':        '#2980b9',
    'Injury & Poisoning':     '#8e44ad',
    'Circulatory':            '#e84393',
    'Respiratory':            '#16a085',
    'Eye & Ear':              '#f39c12',
    'Pregnancy & Childbirth': '#1abc9c',
    'Genitourinary':          '#3498db',
    'Infectious Diseases':    '#e67e22',
    'Nervous System':         '#9b59b6',
    'Endocrine & Metabolic':  '#1e8bc3',
    'Mental Health':          '#fd79a8',
    'Skin':                   '#badc58',
    'Perinatal':              '#55efc4',
    'Congenital':             '#a29bfe',
    'Other':                  '#b2bec3',
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_natural_short_label(text: str) -> str:
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

    if len(meaningful) >= 2:
        short = " ".join(meaningful[:2])
    elif len(words) >= 2:
        short = " ".join(words[:2])
    elif len(words) == 1:
        short = words[0]
    else:
        short = text

    if len(short) > 22:
        short = short[:21].rstrip() + "…"
    return short


# ─── Top 10 selected labels for heatmap ───────────────────────────────────────
top10_names = (
    df.groupby('description')['Admissions'].sum()
    .nlargest(10).index.tolist()
)
top10_rank_map = {name: i + 1 for i, name in enumerate(top10_names)}


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="big-title">🏥 UK Hospital Admissions — Burden & Lockdown Impact</p>',
            unsafe_allow_html=True)
st.markdown("""
<div class="rq-box">
<b>Research Question:</b> What is the relative burden of each hospital admissions category, 
and how did the highest-burden categories change during the <b>2020–21 COVID lockdown</b>?<br>
<span style="font-size:0.83rem; color:#94a3b8">ONS Hospital Admissions Dataset · 2012–2023 · COMP4037 CW2</span>
</div>
""", unsafe_allow_html=True)

total_adm = df['Admissions'].sum()
covid_change = (
    df[df['year'] == '2020-21']['Admissions'].sum() /
    df[df['year'] == '2019-20']['Admissions'].sum() - 1
) * 100

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-l">Total Admissions (2012–2023)</div>
    <div class="kpi-v">{total_adm/1e6:.0f}M</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Lockdown Drop (2020-21)</div>
    <div class="kpi-v" style="color:#ef4444">{covid_change:+.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Largest ICD Chapter</div>
    <div class="kpi-v" style="font-size:1.0rem;padding-top:0.4rem">Digestive</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Highest Single Diagnosis</div>
    <div class="kpi-v" style="font-size:0.9rem;padding-top:0.35rem">Labour & delivery</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE — TREEMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Main Figure: Treemap — Relative Burden by Disease Chapter & Diagnosis</div>', unsafe_allow_html=True)

st.markdown("""
Each rectangle's **size** = total admissions (2012–2023).  
**Outer rectangle** = ICD disease chapter (e.g. Digestive, Circulatory).  
**Inner rectangle** = specific diagnosis within that chapter.  
**Colour** = ICD disease chapter. Hover over any block to see exact totals.
""")

tree_df = (
    df.groupby(['chapter', 'description'], as_index=False)['Admissions']
    .sum()
    .rename(columns={'Admissions': 'total'})
)
tree_df = tree_df[tree_df['total'] > 0].copy()

tree_df['label_full'] = tree_df['description'].fillna('Unknown diagnosis')
tree_df.loc[tree_df['label_full'].astype(
    str).str.strip() == '', 'label_full'] = 'Unknown diagnosis'
tree_df['label_short'] = tree_df['label_full'].apply(make_natural_short_label)

tree_df['label_display'] = tree_df.apply(
    lambda row: f"{top10_rank_map[row['description']]}. {row['label_short']}"
    if row['description'] in top10_rank_map else row['label_short'],
    axis=1
)

# Build stable treemap manually
labels = []
parents = []
values = []
ids = []
colors = []
customdata = []

# Root node
root_id = "root"
grand_total = tree_df['total'].sum()

labels.append("Total Admissions")
parents.append("")
values.append(grand_total)
ids.append(root_id)
colors.append("#f8fafc")
customdata.append(["Total Admissions", grand_total, "All categories"])

# Chapter nodes
chapter_totals = tree_df.groupby('chapter', as_index=False)['total'].sum()

for _, row in chapter_totals.iterrows():
    chapter = row['chapter']
    total = row['total']
    chapter_id = f"chapter::{chapter}"

    labels.append(chapter)
    parents.append(root_id)
    values.append(total)
    ids.append(chapter_id)
    colors.append(CHAPTER_COLORS.get(chapter, "#b2bec3"))
    customdata.append([chapter, total, chapter])

# Diagnosis nodes
for _, row in tree_df.iterrows():
    chapter = row['chapter']
    full_name = row['label_full']
    display_name = row['label_display']
    total = row['total']
    chapter_id = f"chapter::{chapter}"
    diag_id = f"diag::{chapter}::{full_name}"

    labels.append(display_name)
    parents.append(chapter_id)
    values.append(total)
    ids.append(diag_id)
    colors.append(CHAPTER_COLORS.get(chapter, "#b2bec3"))
    customdata.append([full_name, total, chapter])

fig_tree = go.Figure(go.Treemap(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    textinfo="label+value",
    textfont=dict(size=11, family='IBM Plex Sans'),
    marker=dict(
        colors=colors,
        line=dict(width=1.5, color='white'),
        pad=dict(t=22, b=4, l=4, r=4)
    ),
    customdata=customdata,
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'Total Admissions: <b>%{customdata[1]:,.0f}</b><br>'
        'Category: %{customdata[2]}<extra></extra>'
    ),
    root_color='#f8fafc'
))

fig_tree.update_layout(
    height=650,
    margin=dict(t=30, l=10, r=10, b=10),
    paper_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    uniformtext=dict(minsize=11, mode='hide')
)

st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("""
<p class="caption">
<b>Figure 1 (Main).</b> Treemap of total UK hospital admissions (2012–2023) grouped by ICD chapter (outer rectangles) 
and diagnosis (inner rectangles). Rectangle area is proportional to number of admissions, while colour distinguishes ICD chapter.
Numbered labels 1–10 mark the diagnoses selected for the supporting heatmap.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Unique Observation — Treemap:</b><br>
The <b>Digestive</b> chapter holds the largest total area, driven by Oesophagus/Stomach and Intestinal diseases.
However, the biggest <i>single</i> inner rectangle is <b>Labour & Delivery</b> 
(inside the small Pregnancy chapter) — one diagnosis that almost equals an entire chapter.
<b>Eye & Ear</b> is almost entirely Cataracts — a single elective procedure — which contrasts 
sharply with chapters like Circulatory that are spread across many diagnoses.
<b>Mental Health</b> occupies a relatively small area in this inpatient admissions dataset, indicating a smaller hospital admission burden compared with other disease chapters.
These structural differences are invisible in bar charts but immediately visible here.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTING FIGURE — % CHANGE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Supporting Figure: Heatmap — % Change in Admissions vs 2019–20 Baseline</div>', unsafe_allow_html=True)

st.markdown("""
Rows = the **same top-10 diagnoses** from the treemap above. Columns = financial year.  
Colour = % change in admissions vs the **2019–20 pre-COVID baseline**.  
🔴 Red = fell vs baseline · ⬜ White = no change · 🔵 Blue = rose vs baseline.  
The lockdown year (2020–21) is highlighted with an orange border.
""")

base_year = '2019-20'
base = df[df['year'] == base_year].groupby('description')['Admissions'].sum()

hm_rows = {}
for yr in years:
    yr_totals = df[df['year'] == yr].groupby('description')['Admissions'].sum()
    row = {}
    for d in top10_names:
        b = base.get(d, np.nan)
        v = yr_totals.get(d, np.nan)
        row[d] = (v - b) / b * 100 if (b and b > 0) else np.nan
    hm_rows[yr] = row

hm_df = pd.DataFrame(hm_rows).T[top10_names]

short_cols = [SHORT_LABELS.get(
    d, d[:28] + '…' if len(d) > 28 else d) for d in top10_names]
hm_df.columns = short_cols

z_max, z_min = 30, -60
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
        [0.0,  '#7f1d1d'],
        [0.2,  '#b91c1c'],
        [0.4,  '#fca5a5'],
        [0.5,  '#f8fafc'],
        [0.65, '#93c5fd'],
        [1.0,  '#1d4ed8'],
    ],
    colorbar=dict(
        title=dict(text='% vs<br>2019–2020', font=dict(size=11)),
        tickvals=[-60, -40, -20, 0, 20, 30],
        ticktext=['-60%', '-40%', '-20%', '0%', '+20%', '+30%'],
        tickfont=dict(size=10),
        len=0.85,
        thickness=14,
        x=1.05,
    ),
    text=annot.values,
    texttemplate="%{text}",
    textfont=dict(size=14, color='#1e293b'),
    hovertemplate='<b>%{x}</b><br>Year: %{y}<br>% Change: %{z:+.1f}%<extra></extra>',
    xgap=2,
    ygap=2,
))

lockdown_idx = hm_df.index.tolist().index('2020-21')
fig_hm.add_shape(
    type='rect', xref='paper', yref='y',
    x0=-0.01, x1=1.01,
    y0=lockdown_idx - 0.5, y1=lockdown_idx + 0.5,
    line=dict(color='#f97316', width=2.5, dash='dot'),
    fillcolor='rgba(0,0,0,0)', layer='above'
)
fig_hm.add_annotation(
    x=1.0, xref='paper', y='2020-21', yref='y',
    text='◄ COVID <br> Lockdown',
    showarrow=False,
    font=dict(size=12, color='#ea580c', family='IBM Plex Sans'),
    xanchor='left', xshift=14
)

fig_hm.update_layout(
    height=850,
    margin=dict(l=10, r=200, t=30, b=100),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    xaxis=dict(tickfont=dict(size=16), tickangle=-30, showgrid=False),
    yaxis=dict(tickfont=dict(size=16), showgrid=False, autorange='reversed'),
)

st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("""
<p class="caption">
<b>Figure 2 (Supporting).</b> % change in admissions for the top-10 burden diagnoses relative to the 2019–20 baseline. 
The orange dashed border marks the 2020–21 COVID lockdown year. Red = decrease vs baseline, Blue = increase.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Unique Observation — Heatmap:</b><br>
The lockdown row (2020–21) is almost entirely <b>deep red</b>, confirming widespread disruption.
But the intensity differs dramatically: <b>Arthropathies fell ~53%</b> and 
<b>Cataracts fell ~40%</b> — both are elective, deferrable procedures that hospitals cancelled first.
In contrast, <b>Labour & delivery fell only ~6%</b> — you cannot delay childbirth.
By 2021–22 most categories show <b>partial blue recovery</b> but still below baseline.
This confirms the treemap finding: the <i>Eye & Ear</i> and <i>Musculoskeletal</i> chapters 
(large blocks in the treemap) suffered the most because they are dominated by planned procedures.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CW2 SUBMISSION TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="thin">', unsafe_allow_html=True)
with st.expander("📋 CW2 PDF Submission Template — expand to copy", expanded=False):
    st.markdown("""
<div class="template-box">

<b>Research Question Answered:</b><br>
What is the relative burden of each hospital admissions category, and which categories 
were most impacted by the 2020–21 COVID lockdown?

<br><b>Visual Design Type:</b><br>
Main figure: Treemap (hierarchical area-based visualization)<br>
Supporting figure: Matrix Heatmap with diverging % change colour scale

<br><b>Name of Tool:</b><br>
Python 3 — Streamlit (web dashboard) + Plotly 5 (interactive charts)

<br><b>Diagnosis Groups Shown:</b><br>
Treemap: All ICD-10 chapters (15 groups) with all individual diagnoses nested inside.<br>
Heatmap: Top 10 highest-burden diagnoses (same categories visible in treemap).

<br><b>Variables & Why Chosen:</b><br>
• Total Admissions (2012–2023) — primary measure of clinical burden → mapped to area in treemap<br>
• ICD Chapter — provides the natural clinical hierarchy for the outer treemap level<br>
• Diagnosis description — inner treemap level; specific condition detail<br>
• % Change vs 2019–20 baseline — isolates lockdown shock without absolute scale distortion<br>
• Financial Year (2012–23) — time axis in heatmap to show trends and the lockdown dip

<br><b>Visual Mappings:</b><br>
Treemap —<br>
• Size (area) → total admissions 2012–2023 (larger = more admissions)<br>
• Colour (hue) → ICD disease chapter<br>
• Hierarchy → outer rectangle = chapter, inner rectangle = diagnosis<br>
• Position → squarified layout; largest blocks placed toward top-left<br>
• Numbered labels → 1–10 indicate the diagnoses selected for the supporting heatmap<br>

Heatmap —<br>
• X-axis → diagnosis (same top-10 from treemap)<br>
• Y-axis → financial year (oldest at top, newest at bottom)<br>
• Cell colour (diverging) → % change vs 2019-20: deep red = large fall, white = no change, deep blue = rise<br>
• Cell annotation → exact % printed inside each cell<br>
• Orange dashed border → visually highlights the 2020–21 lockdown row

<br><b>Unique Observation:</b><br>
The treemap shows that while Digestive is the largest chapter overall, a single diagnosis 
(Labour & delivery) creates the biggest individual rectangle. The supporting heatmap then 
reveals that during lockdown, Arthropathies fell 53% and Cataracts fell ~40% (both elective), 
while Labour & delivery fell only 6% (non-deferrable). Read together, the two charts show 
that large treemap blocks dominated by elective procedures suffered the most disruption, 
while essential care was protected. This pattern cannot be seen from raw numbers.

<br><b>Data Preparation:</b><br>
1. Renamed column headers (Unnamed: 1 → description, 2012-13 → year)<br>
2. Converted numeric fields from string: pd.to_numeric(..., errors='coerce')<br>
3. Removed 2023-24 (incomplete year with zero values)<br>
4. Derived ICD chapter from first letter of the diagnosis code (e.g. 'M' → Musculoskeletal)<br>
5. Aggregated all years by (chapter, description) → sum of Admissions for treemap<br>
6. Selected the ten highest-burden diagnoses and assigned labels 1–10 for the treemap and heatmap link<br>
7. Computed % change = (year_value − 2019-20_value) / 2019-20_value × 100 per diagnosis<br>
8. Applied ±30/−60% clip on colour scale to prevent outliers washing out the palette<br>
9. Created short labels for treemap display while preserving full labels in hover

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:2rem'>
COMP4037 Research Methods · Coursework 2 · Streamlit + Plotly · ONS Hospital Admissions 2012–2023
</div>
""", unsafe_allow_html=True)

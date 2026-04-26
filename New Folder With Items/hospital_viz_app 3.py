"""
COMP4037 CW2 — UK Hospital Admissions Visualization
Research Question: What is the relative burden of each hospital admissions category,
and how did high-burden categories change during the 2020-21 COVID lockdown?

Visual Design: Treemap (main) + % Change Heatmap (supporting)
Tool: Python — Streamlit + Plotly
"""

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

.legend-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 0.95rem;
    margin-top: 0.35rem;
}
.legend-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1rem;
    color: #0f172a;
    margin-bottom: 0.6rem;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.35rem;
}
.legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid #cbd5e1;
    flex-shrink: 0;
}
.legend-text {
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.2;
}
.legend-note {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.35rem;
    line-height: 1.35;
}
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

    num_cols = [
        'Admissions', 'Emergency', 'Male', 'Female',
        'Waiting list', 'Planned', 'Mean age',
        'Mean length of stay', 'Mean time waited',
        'Finished consultant episodes', 'FCE bed days'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['year'] != '2023-24']
    df = df.dropna(subset=['year', 'description'])
    df['description'] = df['description'].str.strip()

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
    return df


df = load_data()
years = sorted(df['year'].dropna().unique().tolist())

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
    'Other forms of heart disease': 'Heart disease',
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
def compact_treemap_label(text, max_len=18):
    if pd.isna(text):
        return ""
    text = str(text).strip()

    text = SHORT_LABELS.get(text, text)

    replacements = {
        "Malignant neoplasms": "Cancers",
        "Malignant neoplasm": "Cancer",
        "Disorders": "Disorders",
        "Diseases": "Diseases",
        "Symptoms": "Symptoms",
        "including": "incl.",
        "circumstances related to": "related to",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if len(text) > max_len:
        text = text[:max_len - 1].rstrip() + "…"

    return text


def render_treemap_legend():
    st.markdown('<div class="legend-card">', unsafe_allow_html=True)
    st.markdown('<div class="legend-title">Treemap Legend</div>',
                unsafe_allow_html=True)
    for chapter, color in CHAPTER_COLORS.items():
        st.markdown(
            f"""
            <div class="legend-row">
                <div class="legend-swatch" style="background:{color};"></div>
                <div class="legend-text">{chapter}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown(
        '<div class="legend-note">Rectangle area = total admissions. Colour = ICD chapter.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_heatmap_legend():
    st.markdown('<div class="legend-card">', unsafe_allow_html=True)
    st.markdown('<div class="legend-title">Heatmap Legend</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="legend-row">
            <div class="legend-swatch" style="background:#b91c1c;"></div>
            <div class="legend-text">Red = below 2019–20 baseline</div>
        </div>
        <div class="legend-row">
            <div class="legend-swatch" style="background:#f8fafc;"></div>
            <div class="legend-text">White = around baseline</div>
        </div>
        <div class="legend-row">
            <div class="legend-swatch" style="background:#1d4ed8;"></div>
            <div class="legend-text">Blue = above baseline</div>
        </div>
        <div class="legend-row">
            <div class="legend-swatch" style="background:#ffffff; border:2px dashed #f97316;"></div>
            <div class="legend-text">Orange dashed border = 2020–21 lockdown</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


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
covid_base = df[df['year'] == '2019-20']['Admissions'].sum()
covid_year = df[df['year'] == '2020-21']['Admissions'].sum()
covid_change = ((covid_year / covid_base) - 1) * \
    100 if covid_base > 0 else np.nan

largest_chapter = (
    df.groupby('chapter')['Admissions'].sum(
    ).sort_values(ascending=False).index[0]
    if not df.empty else "N/A"
)
largest_diag = (
    df.groupby('description')['Admissions'].sum(
    ).sort_values(ascending=False).index[0]
    if not df.empty else "N/A"
)
largest_diag_label = SHORT_LABELS.get(
    largest_diag,
    largest_diag[:24] + '…' if isinstance(largest_diag,
                                          str) and len(largest_diag) > 26 else largest_diag
)

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-l">Total Admissions (all years)</div>
    <div class="kpi-v">{total_adm/1e6:.0f}M</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Lockdown Drop (2020-21)</div>
    <div class="kpi-v" style="color:#ef4444">{covid_change:+.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Largest ICD Chapter</div>
    <div class="kpi-v" style="font-size:1.0rem;padding-top:0.4rem">{largest_chapter}</div>
  </div>
  <div class="kpi">
    <div class="kpi-l">Highest Single Diagnosis</div>
    <div class="kpi-v" style="font-size:0.9rem;padding-top:0.35rem">{largest_diag_label}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE — TREEMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Main Figure: Treemap — Relative Burden by Disease Chapter & Diagnosis</div>', unsafe_allow_html=True)

st.markdown("""
Each rectangle's **size** = total admissions (2012–2023).  
**Outer rectangle** = ICD disease chapter.  
**Inner rectangle** = specific diagnosis within that chapter.  
**Colour** = disease chapter. Hover over any block to see the full diagnosis name and exact total.
""")

tree_df = (
    df.groupby(['chapter', 'description'], as_index=False)['Admissions']
    .sum()
    .rename(columns={'Admissions': 'total'})
)
tree_df = tree_df[tree_df['total'] > 0].copy()

tree_df['display_name'] = tree_df['description'].fillna('Unknown diagnosis')
tree_df.loc[tree_df['display_name'].astype(
    str).str.strip() == '', 'display_name'] = 'Unknown diagnosis'
tree_df['display_short'] = tree_df['display_name'].apply(
    lambda x: compact_treemap_label(x, max_len=18))

fig_tree = px.treemap(
    tree_df,
    path=['chapter', 'display_short'],
    values='total',
    color='chapter',
    color_discrete_map=CHAPTER_COLORS,
    custom_data=['display_name', 'total', 'chapter']
)

fig_tree.update_traces(
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'ICD Chapter: %{customdata[2]}<br>'
        'Total Admissions (2012–23): <b>%{customdata[1]:,.0f}</b><extra></extra>'
    ),
    texttemplate='%{label}',
    textfont=dict(size=11, family='IBM Plex Sans', color='white'),
    textinfo='label',
    marker=dict(
        line=dict(width=1.5, color='white'),
        pad=dict(t=22, b=4, l=4, r=4)
    ),
    root_color='#f8fafc'
)

fig_tree.update_layout(
    height=760,
    margin=dict(t=10, l=5, r=5, b=5),
    paper_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    uniformtext=dict(minsize=11, mode='show')
)

tree_col, tree_leg = st.columns([6.8, 1.2], gap="medium")
with tree_col:
    st.plotly_chart(fig_tree, use_container_width=True)
with tree_leg:
    render_treemap_legend()

st.markdown("""
<p class="caption">
<b>Figure 1 (Main).</b> Treemap of total UK hospital admissions (2012–2023) grouped by ICD chapter (outer rectangles) 
and diagnosis (inner rectangles). Rectangle area is proportional to number of admissions. 
Colour identifies the disease chapter.
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
<b>Mental Health</b> is a tiny block, indicating a relatively smaller inpatient hospital burden in this dataset.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTING FIGURE — % CHANGE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Supporting Figure: Heatmap — % Change in Admissions vs 2019–20 Baseline</div>', unsafe_allow_html=True)

st.markdown("""
Rows = financial year. Columns = the top-10 highest-burden diagnoses from the treemap.  
Colour = % change in admissions vs the **2019–20 pre-COVID baseline**.  
🔴 Red = fell vs baseline · ⬜ White = no change · 🔵 Blue = rose vs baseline.
""")

top10_names = (
    df.groupby('description')['Admissions'].sum()
    .nlargest(10).index.tolist()
)

base_year = '2019-20'
base = df[df['year'] == base_year].groupby('description')['Admissions'].sum()

hm_rows = {}
for yr in years:
    yr_totals = df[df['year'] == yr].groupby('description')['Admissions'].sum()
    row = {}
    for d in top10_names:
        b = base.get(d, np.nan)
        v = yr_totals.get(d, np.nan)
        row[d] = (v - b) / b * \
            100 if pd.notna(b) and b > 0 and pd.notna(v) else np.nan
    hm_rows[yr] = row

hm_df = pd.DataFrame(hm_rows).T[top10_names]

short_cols = [
    SHORT_LABELS.get(d, d[:28] + '…' if len(d) > 28 else d)
    for d in top10_names
]
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
        title=dict(text='% vs<br>2019–20', font=dict(size=11)),
        tickvals=[-60, -40, -20, 0, 20, 30],
        ticktext=['-60%', '-40%', '-20%', '0%', '+20%', '+30%'],
        tickfont=dict(size=10),
        len=0.85,
        thickness=14,
        x=1.12,
    ),
    text=annot.values,
    texttemplate="%{text}",
    textfont=dict(size=10.5, color='#1e293b'),
    hovertemplate='<b>%{x}</b><br>Year: %{y}<br>% Change: %{z:+.1f}%<extra></extra>',
    xgap=2,
    ygap=2,
))

if '2020-21' in hm_df.index.tolist():
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
        font=dict(size=10.5, color='#ea580c', family='IBM Plex Sans'),
        xanchor='left', xshift=8
    )

fig_hm.update_layout(
    height=380,
    margin=dict(l=10, r=130, t=15, b=12),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    xaxis=dict(tickfont=dict(size=9.8), tickangle=-30, showgrid=False),
    yaxis=dict(tickfont=dict(size=10.5), showgrid=False, autorange='reversed'),
    uniformtext=dict(minsize=10, mode='hide')
)

hm_col, hm_leg = st.columns([6.8, 1.2], gap="medium")
with hm_col:
    st.plotly_chart(fig_hm, use_container_width=True)
with hm_leg:
    render_heatmap_legend()

st.markdown("""
<p class="caption">
<b>Supporting Figure.</b> % change in admissions for the top-10 burden diagnoses relative to the 2019–20 baseline. 
The orange dashed border marks the 2020–21 COVID lockdown year. Red = decrease vs baseline, Blue = increase.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Unique Observation — Heatmap:</b><br>
The lockdown row (2020–21) is almost entirely <b>deep red</b>, confirming widespread disruption.
But the intensity differs dramatically: <b>Arthropathies fell ~53%</b> and 
<b>Cataracts fell ~46%</b> — both are elective, deferrable procedures that hospitals cancelled first.
In contrast, <b>Labour & delivery fell only ~6%</b> — childbirth-related admissions are less deferrable.
By 2021–22 most categories show <b>partial blue recovery</b> but still below baseline.
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
Treemap: All ICD-10 chapters with all individual diagnoses nested inside.<br>
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
• Label → shortened label in the box; full diagnosis shown on hover<br>

Heatmap —<br>
• X-axis → diagnosis (same top-10 from treemap)<br>
• Y-axis → financial year (oldest at top, newest at bottom)<br>
• Cell colour (diverging) → % change vs 2019-20: deep red = large fall, white = no change, deep blue = rise<br>
• Cell annotation → exact % printed inside each cell<br>
• Orange dashed border → visually highlights the 2020–21 lockdown row

<br><b>Data Preparation:</b><br>
1. Renamed raw columns for clarity<br>
2. Converted numeric fields from string to numeric<br>
3. Removed incomplete year 2023–24<br>
4. Derived ICD chapter from first letter of the diagnosis code<br>
5. Aggregated admissions by chapter and diagnosis for treemap<br>
6. Computed % change relative to 2019–20 for heatmap<br>
7. Created compact label versions for treemap readability while preserving full names in hover

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:2rem'>
COMP4037 Research Methods · Coursework 2 · Streamlit + Plotly · ONS Hospital Admissions 2012–2023
</div>
""", unsafe_allow_html=True)

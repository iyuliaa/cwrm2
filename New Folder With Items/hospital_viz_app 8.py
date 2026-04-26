"""
COMP4037 CW2 — UK Hospital Admissions Visualization
Research Question: What is the relative burden of hospital admission categories across ICD
chapters and diagnoses, and how did the most substantively important diagnoses deviate from
their pre-COVID baseline during the 2020–2021 lockdown and subsequent recovery period?

Visual Design: Hierarchical Treemap (main) + Matrix Heatmap with diverging colour scale (supporting)
Tool: Python 3, Pandas, Plotly, and Streamlit

MODIFICATION v10:
- Top-10 burden diagnoses in treemap now marked with ★ rank badges (ranked #1–#10)
- Treemap labels for top-10 show rank prefix: e.g. "★#1 Labour & Delivery"
- Heatmap now uses all TOP-10 BURDEN diagnoses (instead of 5+5 split) to make the
  relationship with the treemap explicit and academically stronger
- Bridge section added between figures to explain the visual narrative link
- Top-10 legend added below treemap
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
.top10-legend {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    margin: 0.6rem 0 1rem 0;
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.7;
}
.top10-legend b { color: #0f172a; }
.rank-badge {
    display: inline-block;
    background: #fef08a;
    border: 1.5px solid #eab308;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #713f12;
    margin-right: 4px;
}
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
    'Complications of labour and delivery': 'Labour complications',
    'Labour and delivery': 'Labour & Delivery',
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
    'Digestive':              '#d97706',
    'Symptoms & Signs':       '#b45309',
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
    'Nervous System':         '#9333ea',
    'Endocrine & Metabolic':  '#0369a1',
    'Mental Health':          '#be185d',
    'Skin':                   '#65a30d',
    'Perinatal':              '#059669',
    'Congenital':             '#6d28d9',
    'Other':                  '#94a3b8',
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_short_label(text: str) -> str:
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


# ─── Compute Top-10 Burden (for treemap markers AND heatmap) ──────────────────
# This is the KEY link: the same top-10 used to mark treemap are the columns in heatmap
diag_burden = df.groupby('description')[
    'Admissions'].sum().sort_values(ascending=False)
top10_burden_list = diag_burden.nlargest(
    10).index.tolist()   # ranked #1 to #10
top10_burden_set = set(top10_burden_list)

# rank dict: description → rank (1 = highest)
top10_rank = {d: i+1 for i, d in enumerate(top10_burden_list)}


# ─── Heatmap: use ALL top-10 burden diagnoses (explicit link to treemap) ──────
base_ser = df[df['year'] ==
              '2019-20'].groupby('description')['Admissions'].sum()
lock_ser = df[df['year'] ==
              '2020-21'].groupby('description')['Admissions'].sum()


def lockdown_pct(d):
    b = base_ser.get(d, np.nan)
    l = lock_ser.get(d, np.nan)
    return (l - b) / b * 100 if (pd.notna(b) and b > 0) else 0


# Order heatmap columns: largest lockdown drop → most resilient (left → right)
selected_diagnoses_sorted = sorted(top10_burden_list, key=lockdown_pct)


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
**★ #1–#10** = Top-10 highest-burden diagnoses (ranked by total admissions) — these are the diagnoses examined in Figure 2.  
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

# Visibility threshold
grand_total = tree_df['total'].sum()
chapter_totals = tree_df.groupby('chapter')['total'].sum()
MIN_DIAG_FRACTION = 0.008

tree_df['show_label'] = tree_df['total'] / grand_total >= MIN_DIAG_FRACTION
tree_df['is_top10'] = tree_df['label_full'].isin(top10_burden_set)

# Build label: top-10 get "★#N Label", others get plain short label or ""


def build_display_label(row):
    if row['is_top10']:
        rank = top10_rank[row['label_full']]
        base = row['label_short']
        return f"★#{rank} {base}"
    elif row['show_label']:
        return row['label_short']
    else:
        return ""


tree_df['label_display'] = tree_df.apply(build_display_label, axis=1)

# Top-10 diagnoses always show their label regardless of size threshold
tree_df.loc[tree_df['is_top10'], 'label_display'] = tree_df[tree_df['is_top10']].apply(
    lambda r: f"★#{top10_rank[r['label_full']]} {r['label_short']}", axis=1
)

# Build manual treemap structure
labels, parents, values, ids, colors, customdata, textcolors = [], [], [], [], [], [], []

root_id = "root"
labels.append("All Admissions")
parents.append("")
values.append(grand_total)
ids.append(root_id)
colors.append("#f1f5f9")
customdata.append(["All Admissions", grand_total, "—", ""])
textcolors.append("white")

for chapter, ch_total in chapter_totals.sort_values(ascending=False).items():
    chapter_id = f"ch::{chapter}"
    labels.append(chapter)
    parents.append(root_id)
    values.append(ch_total)
    ids.append(chapter_id)
    colors.append(CHAPTER_COLORS.get(chapter, "#94a3b8"))
    customdata.append([chapter, ch_total, chapter, ""])
    textcolors.append("white")

for _, row in tree_df.iterrows():
    chapter = row['chapter']
    full_name = row['label_full']
    disp_name = row['label_display']
    total = row['total']
    is_top10 = row['is_top10']
    chapter_id = f"ch::{chapter}"
    diag_id = f"dx::{chapter}::{full_name}"

    if is_top10:
        rank = top10_rank[full_name]
        rank_label = f"★ Top-10 Burden Rank #{rank}"
        # Slightly lighter / highlighted version of chapter colour by mixing with gold
        base_color = CHAPTER_COLORS.get(chapter, "#94a3b8")
        cell_color = base_color   # keep chapter colour; we'll use border + label to distinguish
    else:
        rank_label = ""
        cell_color = CHAPTER_COLORS.get(chapter, "#94a3b8")

    labels.append(disp_name)
    parents.append(chapter_id)
    values.append(total)
    ids.append(diag_id)
    colors.append(cell_color)
    customdata.append([full_name, total, chapter, rank_label])
    # gold text for top-10
    textcolors.append("#fef08a" if is_top10 else "white")

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
        line=dict(
            width=[4 if (i_id.startswith("dx::") and
                         any(full in i_id for full in top10_burden_set))
                   else 2 for i_id in ids],
            color=[
                '#fef08a' if (i_id.startswith("dx::") and
                              any(full in i_id for full in top10_burden_set))
                else 'white'
                for i_id in ids
            ]
        ),
        pad=dict(t=24, b=5, l=5, r=5)
    ),
    customdata=customdata,
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'Total Admissions (2012–2023): <b>%{customdata[1]:,.0f}</b><br>'
        'ICD Chapter: %{customdata[2]}<br>'
        '%{customdata[3]}<extra></extra>'
    ),
    root_color='#f1f5f9'
))

# Patch text colour for top-10 via textfont per-cell workaround
# (Plotly treemap doesn't support per-cell textfont, so we use label prefix ★ as visual cue)

fig_tree.update_layout(
    height=620,
    margin=dict(t=30, l=8, r=8, b=8),
    paper_bgcolor='white',
    font=dict(family='IBM Plex Sans'),
    uniformtext=dict(minsize=9, mode='hide')
)

st.plotly_chart(fig_tree, use_container_width=True)

# ── Top-10 ranked legend below treemap ────────────────────────────────────────
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
<b>★ Top-10 Highest-Burden Diagnoses</b> (marked in treemap · these are the columns examined in Figure 2):<br>
{top10_legend_rows}
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p class="caption">
<b>Figure 1 (Main).</b> Hierarchical treemap of total UK hospital admissions (2012–2023).
Outer rectangles represent ICD disease chapters; inner rectangles represent individual diagnosis groups nested within each chapter.
Rectangle <b>area</b> is proportional to cumulative admissions, making relative burden immediately visible across scales.
<b>Colour</b> encodes ICD chapter categorically. <b>★#1–#10 labels with gold borders</b> identify the ten highest-burden 
diagnoses, which form the basis of Figure 2. Labels are shown only for diagnosis groups 
large enough to be legible; full names and exact counts are available on hover.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Treemap Observation:</b><br>
The <b>Digestive</b> chapter holds the largest cumulative area, indicating the greatest aggregate burden across the study period,
driven by a spread of diagnoses rather than a single dominant condition.
In contrast, <b>Pregnancy &amp; Childbirth</b> is a comparatively small chapter by area, yet it contains the single largest
inner rectangle — <b>★#1 Labour &amp; Delivery</b> — a diagnosis whose admissions alone rival entire mid-sized chapters.
The <b>Eye &amp; Ear</b> chapter is almost entirely composed of one diagnosis (★ Cataracts / Lens Disorders),
illustrating how some chapters are clinically diverse while others are dominated by a single high-volume procedure.
These structural differences in chapter composition are invisible in standard bar charts but immediately apparent here.
<b>The ten ★-marked diagnoses are examined in Figure 2</b> to explore whether their high burden was matched by resilience or vulnerability during the COVID lockdown.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE — Explicit narrative link between treemap and heatmap
# ══════════════════════════════════════════════════════════════════════════════
# Build a mini table of top-10 with their lockdown % for the bridge text
bridge_items = []
for rank, diag in enumerate(top10_burden_list, 1):
    short = SHORT_LABELS.get(diag, make_short_label(diag))
    lp = lockdown_pct(diag)
    bridge_items.append(f"<b>★#{rank} {short}</b> ({lp:+.0f}%)")
bridge_text = " · ".join(bridge_items)

st.markdown(f"""
<div class="bridge-box">
<b>🔗 Link between Figure 1 and Figure 2:</b><br>
Figure 1 (Treemap) answers <i>"which diagnoses carry the greatest cumulative admission burden?"</i>
Figure 2 (Heatmap) directly follows up by asking <i>"how did those exact same top-10 diagnoses
behave over time — particularly during the 2020–21 COVID lockdown?"</i><br><br>
The ten ★-marked diagnoses from the treemap are the ten <b>columns</b> in the heatmap below, ordered from the
diagnosis that suffered the <b>largest lockdown drop</b> (left) to the most <b>resilient</b> (right).
This pairing tests whether high burden translates to resilience, or whether even the busiest diagnostic
categories were disrupted by the pandemic.<br><br>
<span style="font-size:0.83rem">Top-10 burden diagnoses and their 2020–21 lockdown change: {bridge_text}</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORTING FIGURE — % CHANGE HEATMAP (now using all top-10 burden)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-label">Figure 2 (Supporting) — Matrix Heatmap: % Change in Admissions vs 2019–20 Baseline</div>',
    unsafe_allow_html=True
)
st.markdown("""
Columns = **Top-10 highest-burden diagnoses from Figure 1 (★#1–#10)**, ordered by severity of lockdown drop (left = most disrupted).  
Rows = financial year. **Cell colour** = % change vs the 2019–20 pre-COVID baseline.  
🔴 Red = below baseline · ⬜ White = near baseline · 🔵 Blue = above baseline.  
The orange dashed border marks the 2020–21 lockdown year.
""")

# Build heatmap matrix using top-10 burden (sorted by lockdown drop)
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

# Column labels: include rank badge in heatmap column labels


def hm_col_label(d):
    short = SHORT_LABELS.get(d, (d[:24] + '…' if len(d) > 24 else d))
    if d in top10_rank:
        return f"★#{top10_rank[d]} {short}"
    return short


short_cols = [hm_col_label(d) for d in selected_diagnoses_sorted]
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
        [0.00, '#7f1d1d'],
        [0.15, '#b91c1c'],
        [0.35, '#fca5a5'],
        [0.50, '#f8fafc'],
        [0.65, '#93c5fd'],
        [0.85, '#1d4ed8'],
        [1.00, '#1e3a8a'],
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

resilient_col = lockdown_vals.idxmax()
resilient_idx = short_cols.index(resilient_col)
fig_hm.add_annotation(
    x=resilient_idx, xref='x', y=lockdown_idx - 0.65, yref='y',
    text="▲ most<br>resilient",
    showarrow=False,
    font=dict(size=9, color='#1e40af', family='IBM Plex Sans'),
    xanchor='center'
)

post_lockdown_years = hm_df.index.tolist()[lockdown_idx + 1:]
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
        break

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
★#N = Treemap burden rank · Columns ordered: most disrupted → most resilient (left → right)
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p class="caption">
<b>Figure 2 (Supporting).</b> Matrix heatmap showing annual % change in admissions relative to the 2019–20 pre-COVID baseline,
for the <b>ten highest-burden diagnoses identified in Figure 1 (★#1–#10)</b>.
Columns are ordered from largest lockdown drop (left) to most resilient (right) to facilitate visual comparison.
Red = decline vs baseline; white = near-baseline; blue = rise above baseline.
The orange dashed border highlights the 2020–21 COVID lockdown year.
Cell values show exact % change. Analytic annotations mark the largest drop, most resilient, and first rebound above baseline.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="observation-box">
<b>🔍 Cross-visual Observation (Treemap → Heatmap):</b><br>
The treemap reveals that the ten highest-burden diagnoses (★#1–#10) span multiple ICD chapters and vary greatly in their
proportional contribution to admission totals. The heatmap then tests whether this dominance in volume translated to 
resilience during the 2020–21 lockdown. <b>It did not — burden and resilience are not aligned.</b><br><br>
<b>★#1 Labour &amp; Delivery</b> — the single largest block in the treemap — fell by only ~7%, demonstrating that
emergency and time-critical care was largely maintained. In contrast, <b>★#3 Arthropathies</b> and diagnoses dominated
by elective and deferrable procedures dropped by over 50%, despite also ranking among the highest-volume categories.
The heatmap also reveals that recovery after 2020–21 is incomplete and uneven: some categories rebound to or above
baseline by 2021–22, while others remain persistently below — indicating structural service disruption rather than
a simple one-year shock. <b>Together, the two visuals show that size in the treemap predicts neither vulnerability
nor resilience during the pandemic; the heatmap is essential to complete the picture.</b>
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
The treemap shows all ICD disease chapters (outer hierarchy) and diagnosis groups nested within each chapter (inner hierarchy).
The ten highest-burden diagnoses (★#1–#10, ranked by total cumulative admissions 2012–2023) are explicitly marked with
a ★ rank badge and gold-bordered labels in the treemap. These exact ten diagnoses form the ten columns of the supporting heatmap,
creating a direct visual and analytical link between the two figures.

<br><br>
<b>Variables:</b><br>
• <b>Total admissions</b> — mapped to rectangle area in the treemap; represents cumulative hospital burden.<br>
• <b>ICD chapter</b> — mapped to colour in the treemap; provides the top-level hierarchy.<br>
• <b>Diagnosis description</b> — identifies specific admission categories; forms the columns in the heatmap.<br>
• <b>Financial year</b> — forms the row axis of the heatmap; reveals temporal patterns.<br>
• <b>Percentage change relative to the 2019–20 baseline</b> — mapped to cell colour in the heatmap;
  highlights disruption and recovery relative to pre-COVID levels.

<br><br>
<b>Visual Mappings:</b><br>
<u>Treemap:</u><br>
• Hierarchy → outer rectangles = ICD chapters; inner rectangles = diagnosis groups.<br>
• Area → total admissions (2012–2023); larger = greater burden.<br>
• Colour → ICD chapter (categorical); separates disease families.<br>
• ★ rank label + gold border → marks the ten highest-burden diagnoses ranked #1–#10; these are the bridge to Figure 2.<br>
• Hover → full diagnosis name and exact admission count.<br>

<u>Heatmap:</u><br>
• X-axis → top-10 burden diagnoses from treemap (★#1–#10), ordered from largest lockdown drop to most resilient.<br>
• Y-axis → financial year (oldest to newest, top to bottom).<br>
• Cell colour → % change vs 2019–20 baseline (red = below; white = near; blue = above).<br>
• Dashed orange border → 2020–21 COVID lockdown year.<br>
• Cell annotations → exact % values for precise comparison.<br>
• Analytic annotations → mark largest drop, most resilient, and first rebound above baseline.

<br><br>
<b>Unique Observation:</b><br>
The treemap identifies the ten highest-burden diagnoses across ICD chapters (★#1–#10); the heatmap
tests whether high cumulative burden corresponded to resilience during the COVID lockdown. The answer
is no — burden and resilience are not aligned. Labour and Delivery (★#1, the largest single treemap block)
fell by only ~7% in 2020–21, as emergency and time-critical care was maintained. In contrast, Arthropathies
(★#3) dropped by over 50% despite its high burden rank, because it encompasses deferrable elective procedures.
Recovery was uneven: some diagnoses rebounded to or above baseline by 2021–22, while others remained
persistently below, indicating structural long-term disruption. The heatmap is therefore not merely
supplementary but analytically necessary: without it, the treemap alone could misleadingly suggest that
the most burdensome diagnoses were the most disrupted.

<br><br>
<b>Data Preparation:</b><br>
The dataset was cleaned by standardising column names, converting numeric fields, removing the partial 2023–24 year,
and dropping rows with missing diagnosis or year values. ICD chapter was derived from the first letter of the
diagnosis code. Admissions were aggregated by chapter and diagnosis across 2012–2023 for the treemap.
Top-10 burden diagnoses were identified by total cumulative admissions and marked in the treemap with ★ rank badges.
These same ten diagnoses form the heatmap columns, ordered by magnitude of 2020–21 lockdown decline.
Annual % change was computed relative to the 2019–20 pre-COVID baseline to normalise comparisons across
diagnoses of different absolute scales. Shortened labels are used for legibility; hover tooltips preserve full names.

<br><br>
<b>Why this design is effective:</b><br>
The treemap compactly encodes the full hierarchy of burden across all ICD chapters and diagnoses in a single view,
with area proportional to cumulative admissions. The ★ rank markers create an explicit visual bridge to Figure 2,
making the analytical connection between the two figures immediately apparent. The heatmap completes the story by
revealing how the top-burden diagnoses behaved over time and during the COVID shock — a temporal dimension that a
treemap cannot express. Together, the two figures answer orthogonal but complementary questions: Figure 1 asks
"how much?" and Figure 2 asks "how resilient?" — neither question can be answered by the other's visual type.

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

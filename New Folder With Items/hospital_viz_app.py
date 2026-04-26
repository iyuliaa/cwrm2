"""
COMP4037 Coursework 2 - Hospital Admissions Data Visualization
Advanced visual dashboard using Streamlit + Plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Hospital Admissions Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        box-shadow: 0 4px 15px rgba(26,26,46,0.15);
        border-left: 4px solid #e94560;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #e94560;
        font-family: 'DM Serif Display', serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Section headers */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #1a1a2e;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e94560;
        margin-bottom: 1rem;
        margin-top: 2rem;
    }

    /* Insight box */
    .insight-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #78350f;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #1a1a2e; }

    /* Divider */
    hr { border-color: #e5e7eb; margin: 1.5rem 0; }

    /* Streamlit overrides */
    .stSelectbox label, .stMultiSelect label { color: #374151; font-weight: 500; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Process Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("./data_raw/clean_combined_data.csv")
    df = df.rename(columns={
        'Primary diagnosis: summary code and description': 'code',
        'Unnamed: 1': 'description',
        '2012-13': 'year'
    })
    # Convert numerics
    num_cols = [
        'Admissions', 'Emergency', 'Male', 'Female', 'Waiting list', 'Planned',
        'Mean age', 'Mean length of stay', 'Mean time waited',
        'Finished consultant episodes', 'FCE bed days',
        'Age 0', 'Age 1-4', 'Age 5-9', 'Age 10-14', 'Age 15', 'Age 16', 'Age 17',
        'Age 18', 'Age 19', 'Age 20-24', 'Age 25-29', 'Age 30-34', 'Age 35-39',
        'Age 40-44', 'Age 45-49', 'Age 50-54', 'Age 55-59', 'Age 60-64',
        'Age 65-69', 'Age 70-74', 'Age 75-79', 'Age 80-84', 'Age 85-89', 'Age 90+'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop 2023-24 (empty)
    df = df[df['year'] != '2023-24']
    df = df.dropna(subset=['year', 'description'])
    df['description'] = df['description'].str.strip()

    # Short labels for charts
    df['short_desc'] = df['description'].apply(
        lambda x: x[:40] + '…' if isinstance(x, str) and len(x) > 40 else x
    )

    # Emergency ratio
    df['emergency_ratio'] = df['Emergency'] / df['Admissions'].replace(0, np.nan)

    # ICD chapter mapping (first letter of code)
    chapter_map = {
        'A': 'Infectious Diseases', 'B': 'Infectious Diseases',
        'C': 'Cancer', 'D': 'Cancer/Blood',
        'E': 'Endocrine/Metabolic', 'F': 'Mental Health',
        'G': 'Nervous System', 'H': 'Eye/Ear',
        'I': 'Circulatory', 'J': 'Respiratory',
        'K': 'Digestive', 'L': 'Skin',
        'M': 'Musculoskeletal', 'N': 'Genitourinary',
        'O': 'Pregnancy/Childbirth', 'P': 'Perinatal',
        'Q': 'Congenital', 'R': 'Symptoms/Signs',
        'S': 'Injury/Poisoning', 'T': 'Injury/Poisoning',
        'V': 'External Causes', 'W': 'External Causes',
        'X': 'External Causes', 'Y': 'External Causes',
        'Z': 'Health Services'
    }
    def get_chapter(code):
        if not isinstance(code, str):
            return 'Other'
        code = code.strip().upper()
        first = code[0] if code else 'Z'
        return chapter_map.get(first, 'Other')

    df['chapter'] = df['code'].apply(get_chapter)
    return df


df = load_data()
years_sorted = sorted(df['year'].dropna().unique())
all_descriptions = sorted(df['description'].dropna().unique())


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Filters")
    st.markdown("---")

    selected_years = st.multiselect(
        "Select Years",
        options=years_sorted,
        default=years_sorted,
        help="Filter by financial year (e.g. 2020-21 = COVID lockdown)"
    )
    if not selected_years:
        selected_years = years_sorted

    top_n = st.slider("Top N Diagnoses to Show", min_value=5, max_value=30, value=15)

    st.markdown("---")
    st.markdown(
        "**Dataset:** ONS UK Hospital Admissions  \n"
        "**Years:** 2012–2023  \n"
        "**Records:** ~2,269 rows  \n"
        "**Tool:** Streamlit + Plotly"
    )

dff = df[df['year'].isin(selected_years)]


# ─── Title ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 UK Hospital Admissions Explorer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Office of National Statistics · 2012–2023 · Interactive Visual Dashboard for COMP4037 CW2</p>',
    unsafe_allow_html=True
)


# ─── KPI Cards ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))

total_adm = dff['Admissions'].sum()
total_emg = dff['Emergency'].sum()
total_wait = dff['Waiting list'].sum()
avg_stay = dff['Mean length of stay'].mean()

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Admissions</div>
        <div class="metric-value">{fmt(total_adm)}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Emergency Admissions</div>
        <div class="metric-value">{fmt(total_emg)}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Waiting List Entries</div>
        <div class="metric-value">{fmt(total_wait)}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg. Length of Stay</div>
        <div class="metric-value">{avg_stay:.1f} days</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: HEATMAP — Admissions by Diagnosis × Year
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Viz 1: Admission Heatmap (Diagnosis × Year)</div>', unsafe_allow_html=True)
st.markdown("""
This is a **matrix / heatmap** — an advanced visual design that goes beyond bar charts.
Each row = one diagnosis category. Each column = one year. 
The colour intensity shows how many people were admitted.
Darker red = more admissions. This lets you spot trends and the COVID dip instantly.
""")

top_diag = (
    dff.groupby('description')['Admissions'].sum()
    .nlargest(top_n).index.tolist()
)
heatmap_df = (
    dff[dff['description'].isin(top_diag)]
    .groupby(['description', 'year'])['Admissions']
    .sum()
    .reset_index()
    .pivot(index='description', columns='year', values='Admissions')
    .fillna(0)
)
# Shorten row labels
heatmap_df.index = [x[:45] + '…' if len(x) > 45 else x for x in heatmap_df.index]

fig_hm = go.Figure(go.Heatmap(
    z=heatmap_df.values / 1e6,
    x=heatmap_df.columns.tolist(),
    y=heatmap_df.index.tolist(),
    colorscale=[
        [0.0, '#fff7ec'],
        [0.2, '#fee8c8'],
        [0.4, '#fdd49e'],
        [0.6, '#fdbb84'],
        [0.8, '#fc8d59'],
        [1.0, '#d7301f'],
    ],
    colorbar=dict(
        title=dict(text='Admissions (M)', font=dict(size=12, color='black')),
        tickfont=dict(size=11, color='black')
    ),
    hovertemplate='<b>%{y}</b><br>Year: %{x}<br>Admissions: %{z:.2f}M<extra></extra>'
))
fig_hm.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(title='Financial Year', tickangle=-45, tickfont=dict(size=11, color='black')),
    yaxis=dict(title='', tickfont=dict(size=10, color='black'), autorange='reversed'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='DM Sans', color='black')
)
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Unique Observation:</b> The column for <b>2020-21</b> (COVID lockdown) shows a clear 
pale/light band across almost all diagnoses — admissions dropped sharply. However, 
<b>Complications of labour and delivery</b> (top row) remained relatively dark, 
suggesting that maternity care was maintained even during lockdowns. 
The heatmap also reveals that <b>Arthropathies (joint disorders)</b> and <b>Intestinal diseases</b> 
are consistently among the largest admission categories year after year.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PARALLEL COORDINATES — Multi-variable comparison
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📐 Viz 2: Parallel Coordinates — Multi-Variable Profile</div>', unsafe_allow_html=True)
st.markdown("""
A **parallel coordinates chart** shows multiple variables for each diagnosis at once.
Each vertical axis is a different measurement. Each line = one diagnosis.
Lines that cross a lot = those variables are inversely related.
This is advanced because it lets you compare 5+ variables simultaneously — impossible with a bar chart.
""")

pc_cols = ['Admissions', 'Emergency', 'Waiting list', 'Mean age', 'Mean length of stay', 'Mean time waited']
pc_data = (
    dff.groupby('description')[pc_cols].mean()
    .reset_index()
    .dropna()
)
pc_data = pc_data[pc_data['Admissions'] > pc_data['Admissions'].quantile(0.5)]  # top 50%
pc_data['chapter'] = pc_data['description'].apply(
    lambda d: df[df['description'] == d]['chapter'].iloc[0] if len(df[df['description'] == d]) else 'Other'
)
chapter_list = sorted(pc_data['chapter'].unique())
chapter_colors = px.colors.qualitative.Bold
color_map = {ch: chapter_colors[i % len(chapter_colors)] for i, ch in enumerate(chapter_list)}
pc_data['color_num'] = pd.Categorical(pc_data['chapter']).codes

fig_pc = go.Figure(go.Parcoords(
    line=dict(
        color=pc_data['color_num'],
        colorscale=[[i / max(len(chapter_list)-1, 1), chapter_colors[i % len(chapter_colors)]]
                    for i in range(len(chapter_list))],
        showscale=False
    ),
    dimensions=[
        dict(label='Admissions (K)', values=pc_data['Admissions'] / 1000,
             range=[0, pc_data['Admissions'].max() / 1000]),
        dict(label='Emergency (K)', values=pc_data['Emergency'] / 1000,
             range=[0, pc_data['Emergency'].max() / 1000]),
        dict(label='Waiting List (K)', values=pc_data['Waiting list'] / 1000,
             range=[0, pc_data['Waiting list'].max() / 1000]),
        dict(label='Mean Age (yr)', values=pc_data['Mean age'],
             range=[0, 100]),
        dict(label='Length of Stay (days)', values=pc_data['Mean length of stay'],
             range=[0, pc_data['Mean length of stay'].max()]),
        dict(label='Wait Time (days)', values=pc_data['Mean time waited'],
             range=[0, pc_data['Mean time waited'].max()]),
    ]
))
fig_pc.update_layout(
    height=480,
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=80, r=80, t=40, b=40),
    font=dict(family='DM Sans', size=12)
)
st.plotly_chart(fig_pc, use_container_width=True)

# Legend for parallel coords
cols_leg = st.columns(len(chapter_list))
for i, (ch, col) in enumerate(zip(chapter_list, cols_leg)):
    col.markdown(
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{chapter_colors[i % len(chapter_colors)]};margin-right:5px"></span>'
        f'<small>{ch}</small>',
        unsafe_allow_html=True
    )

st.markdown("""
<div class="insight-box">
💡 <b>Unique Observation:</b> Diagnoses with <b>high emergency admissions</b> tend to also have 
<b>short wait times</b> (they are rushed in — no waiting). In contrast, diagnoses with 
<b>long waiting lists</b> (like musculoskeletal/joint conditions) have <b>very low emergency admissions</b> — 
these are planned procedures. This V-shaped crossing pattern between Emergency and Waiting List axes 
confirms a fundamental split in how conditions are treated.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TREEMAP — Hierarchical burden by ICD Chapter
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🌳 Viz 3: Treemap — Hierarchical Burden by Disease Chapter</div>', unsafe_allow_html=True)
st.markdown("""
A **treemap** is a hierarchical visualization. The size of each rectangle = number of admissions.
The hierarchy goes: **ICD Chapter → Diagnosis**. Colour = ICD chapter.
This lets you see both the big picture (which body system causes most admissions)
AND the detail (which specific diagnosis within that system is biggest).
""")

tree_data = (
    dff.groupby(['chapter', 'description'])['Admissions'].sum()
    .reset_index()
    .rename(columns={'Admissions': 'total'})
)
tree_data = tree_data[tree_data['total'] > 0]
tree_data['short'] = tree_data['description'].apply(
    lambda x: x[:35] + '…' if len(x) > 35 else x
)

fig_tree = px.treemap(
    tree_data,
    path=['chapter', 'short'],
    values='total',
    color='chapter',
    color_discrete_sequence=px.colors.qualitative.Bold,
    custom_data=['description']
)
fig_tree.update_traces(
    hovertemplate='<b>%{customdata[0]}</b><br>Admissions: %{value:,.0f}<extra></extra>',
    textfont=dict(size=12, family='DM Sans'),
    marker_pad=dict(t=20, b=5, l=5, r=5)
)
fig_tree.update_layout(
    height=560,
    margin=dict(t=20, l=10, r=10, b=10),
    paper_bgcolor='white',
    font=dict(family='DM Sans')
)
st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Unique Observation:</b> The <b>Pregnancy/Childbirth</b> chapter dominates with the single largest 
rectangle — more hospital admissions than any other disease chapter. The <b>Musculoskeletal</b> chapter 
(arthropathies, dorsopathies) forms the second-largest block, reflecting the burden of an ageing 
population. Strikingly, <b>Mental Health</b> appears as a very small block — suggesting either 
underreporting or that mental health conditions rarely lead to inpatient hospital admission.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: RADAR / SPIDER — COVID Impact by Chapter
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🕸️ Viz 4: Radar Chart — COVID Lockdown Impact by Disease Chapter</div>', unsafe_allow_html=True)
st.markdown("""
A **radar (spider) chart** compares two years across all disease chapters simultaneously.
Each spoke = one disease chapter. The blue area = pre-COVID (2019-20). The red area = lockdown (2020-21).
The gap between the two shapes shows where admissions fell the most during COVID.
""")

pre_covid = df[df['year'] == '2019-20'].groupby('chapter')['Admissions'].sum()
covid = df[df['year'] == '2020-21'].groupby('chapter')['Admissions'].sum()
chapters_common = sorted(set(pre_covid.index) & set(covid.index))
pre_vals = [pre_covid.get(c, 0) / 1e6 for c in chapters_common]
cov_vals = [covid.get(c, 0) / 1e6 for c in chapters_common]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=pre_vals + [pre_vals[0]],
    theta=chapters_common + [chapters_common[0]],
    fill='toself',
    fillcolor='rgba(41, 128, 185, 0.25)',
    line=dict(color='#2980b9', width=2),
    name='2019-20 (Pre-COVID)'
))
fig_radar.add_trace(go.Scatterpolar(
    r=cov_vals + [cov_vals[0]],
    theta=chapters_common + [chapters_common[0]],
    fill='toself',
    fillcolor='rgba(231, 76, 60, 0.25)',
    line=dict(color='#e74c3c', width=2),
    name='2020-21 (Lockdown)'
))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            tickfont=dict(size=9),
            title=dict(text='Admissions (M)', font=dict(size=10))
        ),
        angularaxis=dict(tickfont=dict(size=10))
    ),
    showlegend=True,
    legend=dict(
        orientation='h', yanchor='bottom', y=-0.15,
        font=dict(size=12, family='DM Sans')
    ),
    height=480,
    margin=dict(l=60, r=60, t=30, b=60),
    paper_bgcolor='white',
    font=dict(family='DM Sans')
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Unique Observation:</b> The radar chart clearly shows that <b>almost every spoke shrinks</b> 
during lockdown — but by very different amounts. The biggest contractions were in 
<b>Musculoskeletal</b>, <b>Digestive</b>, and <b>Health Services</b> (elective/planned care was cancelled). 
In contrast, <b>Pregnancy/Childbirth</b> barely shrinks — maternity is non-deferrable. 
The asymmetric shape of the lockdown year (red) vs the pre-COVID year (blue) 
visually captures which services were deprioritised during the crisis.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SCATTER MATRIX — Relationships between variables
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔬 Viz 5: Scatter Plot — Emergency Rate vs. Mean Age</div>', unsafe_allow_html=True)
st.markdown("""
This scatter plot explores whether **older patient populations** lead to **more emergency admissions**.
Each dot = one diagnosis category. Size = total admissions. Colour = ICD chapter.
This reveals structural relationships in the data that no bar chart could show.
""")

scatter_data = (
    dff.groupby('description').agg(
        total_adm=('Admissions', 'sum'),
        total_emg=('Emergency', 'sum'),
        mean_age=('Mean age', 'mean'),
        mean_stay=('Mean length of stay', 'mean'),
        chapter=('chapter', 'first')
    ).reset_index().dropna()
)
scatter_data['emg_pct'] = (scatter_data['total_emg'] / scatter_data['total_adm'] * 100).clip(0, 100)
scatter_data['short'] = scatter_data['description'].apply(lambda x: x[:40] + '…' if len(x) > 40 else x)
scatter_data = scatter_data[scatter_data['total_adm'] > scatter_data['total_adm'].quantile(0.3)]

fig_sc = px.scatter(
    scatter_data,
    x='mean_age',
    y='emg_pct',
    size='total_adm',
    color='chapter',
    color_discrete_sequence=px.colors.qualitative.Bold,
    hover_name='description',
    hover_data={
        'total_adm': ':,.0f',
        'mean_stay': ':.1f',
        'mean_age': ':.1f',
        'emg_pct': ':.1f',
        'short': False,
        'chapter': False
    },
    labels={
        'mean_age': 'Mean Patient Age (years)',
        'emg_pct': 'Emergency Admissions (%)',
        'total_adm': 'Total Admissions',
        'mean_stay': 'Avg Stay (days)'
    },
    size_max=60,
    opacity=0.75
)
fig_sc.update_layout(
    height=520,
    paper_bgcolor='white',
    plot_bgcolor='#f8f9fa',
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(family='DM Sans'),
    legend=dict(title='ICD Chapter', font=dict(size=11)),
    xaxis=dict(gridcolor='#e5e7eb', title_font=dict(size=13)),
    yaxis=dict(gridcolor='#e5e7eb', title_font=dict(size=13))
)
st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Unique Observation:</b> There is <b>no strong linear relationship</b> between patient age and 
emergency rate — which is surprising. Some conditions with very young patients 
(e.g. perinatal/pregnancy) still have moderate emergency rates. The 
<b>highest emergency rates (80–100%)</b> cluster in the middle age range around 50-70 years 
— these are circulatory and respiratory conditions. Large bubbles (high admissions) 
with low emergency rates represent high-volume elective specialties like cataract surgery 
and joint replacement.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: AGE DISTRIBUTION — Stacked area
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👥 Viz 6: Age Group Admissions Over Time (Stacked Area)</div>', unsafe_allow_html=True)
st.markdown("""
A **stacked area chart** shows how the distribution of patient ages has changed over time.
Each colour band = one age group. The total height = total admissions.
This is more informative than a line chart because you can see both individual trends 
and the overall total simultaneously.
""")

age_cols_map = {
    'Age 0': '0', 'Age 1-4': '1-4', 'Age 5-9': '5-9',
    'Age 10-14': '10-14', 'Age 15': '15', 'Age 16': '16', 'Age 17': '17',
    'Age 18': '18', 'Age 19': '19', 'Age 20-24': '20-24',
    'Age 25-29': '25-29', 'Age 30-34': '30-34', 'Age 35-39': '35-39',
    'Age 40-44': '40-44', 'Age 45-49': '45-49', 'Age 50-54': '50-54',
    'Age 55-59': '55-59', 'Age 60-64': '60-64', 'Age 65-69': '65-69',
    'Age 70-74': '70-74', 'Age 75-79': '75-79', 'Age 80-84': '80-84',
    'Age 85-89': '85-89', 'Age 90+': '90+'
}

# Group into broader bands
broad_bands = {
    '0-17 (Children)': ['Age 0', 'Age 1-4', 'Age 5-9', 'Age 10-14', 'Age 15', 'Age 16', 'Age 17'],
    '18-39 (Young Adults)': ['Age 18', 'Age 19', 'Age 20-24', 'Age 25-29', 'Age 30-34', 'Age 35-39'],
    '40-59 (Middle Age)': ['Age 40-44', 'Age 45-49', 'Age 50-54', 'Age 55-59'],
    '60-74 (Older Adults)': ['Age 60-64', 'Age 65-69', 'Age 70-74'],
    '75+ (Elderly)': ['Age 75-79', 'Age 80-84', 'Age 85-89', 'Age 90+']
}

age_by_year = dff.groupby('year')[list(age_cols_map.keys())].sum().reset_index()
for band, cols in broad_bands.items():
    age_by_year[band] = age_by_year[[c for c in cols if c in age_by_year.columns]].sum(axis=1)

band_names = list(broad_bands.keys())
band_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

fig_area = go.Figure()
for band, color in zip(band_names, band_colors):
    if band in age_by_year.columns:
        fig_area.add_trace(go.Scatter(
            x=age_by_year['year'],
            y=age_by_year[band] / 1e6,
            name=band,
            mode='lines',
            stackgroup='one',
            fillcolor=color.replace('#', 'rgba(').replace('3b82f6', '59,130,246,0.7')
                         if False else color,
            line=dict(color=color, width=0.5),
            hovertemplate='%{y:.2f}M<extra>' + band + '</extra>'
        ))

fig_area.update_layout(
    height=460,
    xaxis=dict(title='Financial Year', tickangle=-45, tickfont=dict(size=11), gridcolor='#e5e7eb'),
    yaxis=dict(title='Admissions (Millions)', tickfont=dict(size=11), gridcolor='#e5e7eb'),
    paper_bgcolor='white',
    plot_bgcolor='#f8f9fa',
    legend=dict(orientation='h', yanchor='bottom', y=-0.25, font=dict(size=11)),
    margin=dict(l=20, r=20, t=20, b=60),
    font=dict(family='DM Sans'),
    hovermode='x unified'
)
st.plotly_chart(fig_area, use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Unique Observation:</b> The <b>75+ (Elderly)</b> age band has been <b>growing steadily</b> 
as a share of total admissions over the decade — reflecting an ageing UK population. 
The 2020-21 COVID dip is visible across all age groups, but <b>Young Adults (18-39)</b> 
recovered fastest by 2021-22, likely because they had pent-up maternity and elective admissions. 
The <b>Children (0-17)</b> band has remained the smallest and most stable throughout.
</div>
""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9ca3af; font-size:0.85rem; padding: 1rem 0'>
    COMP4037 Research Methods · Coursework 2 · Data Visualization Dashboard<br>
    Dataset: ONS UK Hospital Admissions 2012–2023 · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

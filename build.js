const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat, PageBreak, UnderlineType
} = require('docx');
const fs = require('fs');

// ── Colour palette ──────────────────────────────────────────────────────────
const C = {
  orange:     'F97316',
  darknavy:   '0F172A',
  navy:       '1E3A8A',
  red:        '991B1B',
  slate:      '475569',
  lightslate: '94A3B8',
  white:      'FFFFFF',
  offwhite:   'F8FAFC',
  amber:      'D97706',
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function para(children, opts = {}) {
  return new Paragraph({ children, ...opts });
}
function run(text, opts = {}) {
  return new TextRun({ text, font: 'Georgia', size: 22, ...opts });
}
function bold(text, opts = {}) {
  return run(text, { bold: true, ...opts });
}
function italic(text, opts = {}) {
  return run(text, { italics: true, ...opts });
}
function small(text, opts = {}) {
  return run(text, { size: 19, ...opts });
}

// Section heading with orange underline border
function sectionHead(text) {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C.orange, space: 4 } },
    spacing: { before: 280, after: 100 },
    children: [
      new TextRun({
        text,
        font: 'Arial',
        size: 26,
        bold: true,
        color: C.darknavy,
      })
    ]
  });
}

// Page 1 section title (smaller, for field labels)
function fieldLabel(text) {
  return new Paragraph({
    spacing: { before: 180, after: 40 },
    children: [
      new TextRun({ text, font: 'Arial', size: 22, bold: true, color: C.orange })
    ]
  });
}

function bodyPara(children, spacing = { before: 0, after: 140 }) {
  return new Paragraph({
    spacing,
    alignment: AlignmentType.JUSTIFIED,
    children,
  });
}

function spacer(pts = 160) {
  return new Paragraph({ spacing: { before: 0, after: pts }, children: [] });
}

// Orange horizontal rule via shaded table row (1px line)
function rule() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.orange, space: 2 } },
    spacing: { before: 80, after: 80 },
    children: []
  });
}

// ── Document ─────────────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: 'Georgia', size: 22, color: '1e293b' } }
    },
    paragraphStyles: [
      {
        id: 'Title', name: 'Title', basedOn: 'Normal', quickFormat: true,
        run: { font: 'Arial', size: 40, bold: true, color: C.darknavy },
        paragraph: { spacing: { before: 0, after: 120 }, alignment: AlignmentType.LEFT }
      },
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: 'Arial', size: 26, bold: true, color: C.darknavy },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 0 }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: 'Arial', size: 23, bold: true, color: C.navy },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 }
      },
    ]
  },
  numbering: {
    config: [
      {
        reference: 'bullets',
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: '\u2022',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 600, hanging: 300 } } }
        }]
      }
    ]
  },
  sections: [
    // ══════════════════════════════════════════════════════════════
    // PAGE 1 — Visualizations + Key Metadata
    // ══════════════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 900, right: 900, bottom: 900, left: 900 }
        }
      },
      children: [
        // Title
        new Paragraph({
          style: 'Title',
          spacing: { before: 0, after: 60 },
          children: [
            new TextRun({ text: 'UK Hospital Admissions — Burden & Lockdown Impact', font: 'Arial', size: 34, bold: true, color: C.darknavy })
          ]
        }),

        // Subtitle / module line
        new Paragraph({
          spacing: { before: 0, after: 180 },
          children: [
            new TextRun({ text: 'COMP4037 Research Methods · Coursework 2 · ONS Hospital Admissions Dataset · 2012–2023', font: 'Arial', size: 18, color: C.slate })
          ]
        }),

        rule(),

        // Research Question box (shaded)
        new Table({
          width: { size: 10106, type: WidthType.DXA },
          columnWidths: [10106],
          rows: [
            new TableRow({
              children: [
                new TableCell({
                  shading: { fill: C.darknavy, type: ShadingType.CLEAR },
                  margins: { top: 140, bottom: 140, left: 200, right: 200 },
                  borders: {
                    top: { style: BorderStyle.NONE, size: 0, color: C.darknavy },
                    bottom: { style: BorderStyle.NONE, size: 0, color: C.darknavy },
                    left: { style: BorderStyle.THICK, size: 24, color: C.orange },
                    right: { style: BorderStyle.NONE, size: 0, color: C.darknavy },
                  },
                  children: [
                    new Paragraph({
                      children: [
                        new TextRun({ text: 'Research Question: ', font: 'Arial', size: 20, bold: true, color: 'fb923c' }),
                        new TextRun({ text: 'What is the relative burden of hospital admission categories across ICD chapters and diagnoses, and how did the most substantively important diagnoses deviate from their pre-COVID baseline during the ', font: 'Arial', size: 20, color: 'e2e8f0' }),
                        new TextRun({ text: '2020–21 lockdown', font: 'Arial', size: 20, bold: true, color: 'fb923c' }),
                        new TextRun({ text: ' and subsequent recovery period?', font: 'Arial', size: 20, color: 'e2e8f0' }),
                      ]
                    })
                  ]
                })
              ]
            })
          ]
        }),

        spacer(200),

        // Figure note
        new Paragraph({
          spacing: { before: 0, after: 80 },
          children: [
            new TextRun({ text: 'Figure 1 (Main) — Hierarchical Treemap: ', font: 'Arial', size: 21, bold: true, color: C.darknavy }),
            new TextRun({ text: 'Relative Admission Burden by ICD Chapter & Diagnosis (2012–2023). Rectangle ', font: 'Arial', size: 21, color: C.slate }),
            new TextRun({ text: 'area', font: 'Arial', size: 21, bold: true, color: C.slate }),
            new TextRun({ text: ' ∝ cumulative admissions. ', font: 'Arial', size: 21, color: C.slate }),
            new TextRun({ text: 'Colour', font: 'Arial', size: 21, bold: true, color: C.slate }),
            new TextRun({ text: ' = ICD chapter (categorical). ★#1–#10 = ten highest-burden diagnoses (ranked), forming the column set of Figure 2.', font: 'Arial', size: 21, color: C.slate }),
          ]
        }),

        // Image placeholder notice
        new Table({
          width: { size: 10106, type: WidthType.DXA },
          columnWidths: [10106],
          rows: [
            new TableRow({
              children: [
                new TableCell({
                  shading: { fill: 'F8FAFC', type: ShadingType.CLEAR },
                  margins: { top: 100, bottom: 100, left: 160, right: 160 },
                  borders: {
                    top: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                    bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                    left: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                    right: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                  },
                  children: [
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 280, after: 60 },
                      children: [
                        new TextRun({ text: '[ Figure 1 — Hierarchical Treemap ]', font: 'Arial', size: 22, bold: true, color: C.slate }),
                      ]
                    }),
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 0, after: 60 },
                      children: [new TextRun({ text: 'Insert full-resolution screenshot of the Streamlit treemap here.', font: 'Arial', size: 19, italics: true, color: C.lightslate })]
                    }),
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 0, after: 60 },
                      children: [new TextRun({ text: 'Outer rectangles = ICD chapters  ·  Inner rectangles = diagnoses  ·  ★#1–#10 marked with gold borders and rank labels', font: 'Arial', size: 19, color: C.lightslate })]
                    }),
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 0, after: 280 },
                      children: [new TextRun({ text: 'Largest chapter: Digestive (amber)  ·  Largest single block: ★#1 Labour & Delivery (cyan)  ·  Most concentrated: Eye & Ear (★#6 Cataracts)', font: 'Arial', size: 19, color: C.lightslate })]
                    }),
                  ]
                })
              ]
            })
          ]
        }),

        spacer(140),

        new Paragraph({
          spacing: { before: 0, after: 80 },
          children: [
            new TextRun({ text: 'Figure 2 (Supporting) — Matrix Heatmap: ', font: 'Arial', size: 21, bold: true, color: C.darknavy }),
            new TextRun({ text: '% Change in Admissions vs 2019–20 Baseline · Columns = ★#1–#10 from Figure 1, ordered: most disrupted (left) → most resilient (right).', font: 'Arial', size: 21, color: C.slate }),
          ]
        }),

        new Table({
          width: { size: 10106, type: WidthType.DXA },
          columnWidths: [10106],
          rows: [
            new TableRow({
              children: [
                new TableCell({
                  shading: { fill: 'F8FAFC', type: ShadingType.CLEAR },
                  margins: { top: 100, bottom: 100, left: 160, right: 160 },
                  borders: {
                    top: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                    bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                    left: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                    right: { style: BorderStyle.SINGLE, size: 4, color: 'CBD5E1' },
                  },
                  children: [
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 280, after: 60 },
                      children: [
                        new TextRun({ text: '[ Figure 2 — Matrix Heatmap ]', font: 'Arial', size: 22, bold: true, color: C.slate }),
                      ]
                    }),
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 0, after: 60 },
                      children: [new TextRun({ text: 'Insert full-resolution screenshot of the Streamlit heatmap here.', font: 'Arial', size: 19, italics: true, color: C.lightslate })]
                    }),
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 0, after: 60 },
                      children: [new TextRun({ text: 'Rows = financial years 2012–13 to 2022–23  ·  Columns = ★#1–#10 diagnoses from Figure 1', font: 'Arial', size: 19, color: C.lightslate })]
                    }),
                    new Paragraph({
                      alignment: AlignmentType.CENTER,
                      spacing: { before: 0, after: 280 },
                      children: [new TextRun({ text: 'Diverging scale: deep red (−75%) → white (baseline) → deep blue (+35%)  ·  Orange dashed border = 2020–21 lockdown year', font: 'Arial', size: 19, color: C.lightslate })]
                    }),
                  ]
                })
              ]
            })
          ]
        }),

        spacer(100),

        // Footer note page 1
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 80, after: 0 },
          children: [
            new TextRun({ text: 'Tool: Python 3 · Pandas · Plotly · Streamlit   |   Data: ONS Hospital Admissions 2012–2023   |   COMP4037 CW2', font: 'Arial', size: 17, color: C.lightslate })
          ]
        }),

      ]
    },

    // ══════════════════════════════════════════════════════════════
    // PAGE 2 — Full Description Template
    // ══════════════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1000, right: 1000, bottom: 1000, left: 1000 }
        }
      },
      children: [

        // Page 2 title bar
        new Paragraph({
          spacing: { before: 0, after: 60 },
          children: [
            new TextRun({ text: 'UK Hospital Admissions — Burden & Lockdown Impact', font: 'Arial', size: 30, bold: true, color: C.darknavy }),
          ]
        }),
        new Paragraph({
          spacing: { before: 0, after: 10 },
          children: [
            new TextRun({ text: 'Description Template · COMP4037 CW2', font: 'Arial', size: 19, color: C.lightslate }),
          ]
        }),
        rule(),

        // ── Visual Design Type ──────────────────────────────────
        sectionHead('Visual Design Type'),
        bodyPara([
          run('The submission employs two complementary advanced visual designs. The '),
          bold('primary visual'),
          run(' is a '),
          bold('hierarchical treemap'),
          run(' (Figure 1), which encodes a two-level ICD chapter–diagnosis hierarchy through nested rectangles whose areas are proportional to cumulative hospital admissions (2012–2023). The '),
          bold('supporting visual'),
          run(' is a '),
          bold('matrix heatmap'),
          run(' (Figure 2) with a diverging red–white–blue colour scale, displaying the percentage change in annual admissions relative to the 2019–20 pre-COVID baseline for the ten highest-burden diagnoses identified in Figure 1. Together, the two designs constitute an '),
          italic('orthogonal dual-view'),
          run(': the treemap answers "how much cumulative burden?" while the heatmap answers "how resilient or vulnerable was that burden over time?". Neither question can be answered by the other visual type alone, nor by conventional bar or line charts without losing either the hierarchical composition or the multi-diagnosis temporal matrix.'),
        ]),

        // ── Name of Tool ────────────────────────────────────────
        sectionHead('Name of Tool'),
        bodyPara([
          run('Both figures were produced using '),
          bold('Python 3'),
          run(' with the '),
          bold('Plotly'),
          run(' graphing library ('),
          italic('go.Treemap'),
          run(' and '),
          italic('go.Heatmap'),
          run(' traces) and deployed as an interactive dashboard via '),
          bold('Streamlit'),
          run('. Data wrangling was performed with '),
          bold('Pandas'),
          run(' and '),
          bold('NumPy'),
          run('. All user-interface text is in English. The dashboard supports hover-based tooltips showing exact admission counts, diagnosis full names, and ICD chapter attribution.'),
        ]),

        // ── Variables ───────────────────────────────────────────
        sectionHead('Variables'),
        bodyPara([
          bold('Total Admissions (FAE / Finished Admissions Episodes). '),
          run('The primary quantity of interest, representing the number of completed hospital admission episodes per diagnosis group per financial year. This variable was summed across the full 2012–2023 period to produce a measure of cumulative burden for the treemap, and was also used year-by-year to compute temporal deviation in the heatmap. It was chosen because it directly captures the volume of clinical activity attributable to each diagnostic category.'),
        ]),
        bodyPara([
          bold('ICD-10 Chapter (derived). '),
          run('A categorical variable derived from the leading letter of the WHO ICD-10 diagnosis code supplied in the dataset (e.g., "K" → Digestive, "O" → Pregnancy & Childbirth, "M" → Musculoskeletal). It was constructed to provide a clinically meaningful top-level hierarchy for the treemap, grouping the many hundreds of diagnosis codes into 18 interpretable disease families. Without this derived variable, the treemap would have no valid hierarchy.'),
        ]),
        bodyPara([
          bold('Diagnosis Description. '),
          run('The textual name of each admission category at the summary-code level of ICD-10 abstraction. This variable forms the inner (leaf) level of the treemap hierarchy and defines the columns of the heatmap matrix. Labels were shortened algorithmically (removing stopwords, truncating at 24 characters) for legibility, with full names preserved in interactive hover tooltips.'),
        ]),
        bodyPara([
          bold('Financial Year. '),
          run('The NHS financial year (April–March) in which admissions were recorded, spanning 2012–13 to 2022–23. This forms the row axis of the heatmap, enabling both pre-COVID trends and pandemic-era deviations to be read in temporal sequence across twelve years.'),
        ]),
        bodyPara([
          bold('Percentage Change vs 2019–20 Baseline. '),
          run('A normalised deviation metric computed as ((admissions in year t − admissions in 2019–20) / admissions in 2019–20) × 100. By anchoring all values to the last pre-COVID year, this variable renders the disruption and recovery of diagnoses with very different absolute scales directly comparable within a single colour-encoded matrix. It is the primary variable encoded by cell colour in Figure 2.'),
        ]),

        // ── Visual Mappings ─────────────────────────────────────
        sectionHead('Visual Mappings'),

        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: 'Figure 1 — Hierarchical Treemap', font: 'Arial', size: 22, bold: true, color: C.navy })]
        }),
        bodyPara([
          bold('Hierarchy. '),
          run('The squarified treemap algorithm nests rectangles in two levels: outer rectangles represent ICD chapters (18 disease families); inner rectangles represent individual diagnosis groups aggregated within each chapter. This structure makes both chapter-level composition and diagnosis-level detail simultaneously readable at different scales.'),
        ]),
        bodyPara([
          bold('Area (size). '),
          run('Each rectangle's area is proportional to the total admissions accumulated across 2012–2023 ('),
          italic('branchvalues = "total"'),
          run('). Larger rectangles immediately indicate greater cumulative burden; the Digestive chapter occupies the largest outer block, and Labour & Delivery the largest inner block, at a glance.'),
        ]),
        bodyPara([
          bold('Colour (categorical). '),
          run('Each ICD chapter is assigned a distinct categorical hue drawn from a perceptually varied palette (amber for Digestive, red for Cancer, blue for Health Services, purple for Injury & Poisoning, pink for Circulatory, teal for Respiratory, cyan for Pregnancy & Childbirth, etc.). Colour encodes chapter membership, not admission volume; it separates hierarchical families visually while allowing area alone to convey magnitude. All inner rectangles within a chapter share its chapter colour, reinforcing the two-level structure.'),
        ]),
        bodyPara([
          bold('★ Rank Labels and Gold Borders (special encoding). '),
          run('The ten highest-burden diagnoses (★#1–#10, ranked by total admissions) are distinguished by a gold-bordered cell outline and a rank label prefixed with "★" (e.g., "★#1 Labour & Delivery"). This encoding serves a dual purpose: it allows readers to immediately identify the most clinically significant categories within the treemap, and it creates an explicit visual bridge to Figure 2, whose columns correspond precisely to these ten diagnoses.'),
        ]),
        bodyPara([
          bold('Text Labels. '),
          run('Diagnosis labels are displayed only when the corresponding rectangle exceeds a minimum area threshold (0.8% of the grand total), preventing overprinting in small cells. Full names, exact totals, and chapter attribution are available via interactive hover tooltips.'),
        ]),
        bodyPara([
          bold('Position. '),
          run('Rectangle position is determined by the squarified layout algorithm, which places larger chapters towards the upper-left. While position is not a directly controlled analytical channel, the resulting layout ensures that the highest-burden categories are the most prominent and occupy the most legible screen space.'),
        ]),

        new Paragraph({
          spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: 'Figure 2 — Matrix Heatmap', font: 'Arial', size: 22, bold: true, color: C.navy })]
        }),
        bodyPara([
          bold('X-axis (columns). '),
          run('The ten highest-burden diagnoses from Figure 1, ordered from the largest lockdown drop (leftmost: ★#3 Arthropathies at −53%) to the most resilient (rightmost: ★#1 Labour & Delivery at −6%). This ordering facilitates direct left-to-right comparison of disruption severity.'),
        ]),
        bodyPara([
          bold('Y-axis (rows). '),
          run('Financial years from 2012–13 (top) to 2022–23 (bottom), in chronological order. Reversing the y-axis to descending (oldest top) places pre-COVID years in the upper portion and the pandemic and recovery period in the lower portion, matching the expected reading direction for temporal narratives.'),
        ]),
        bodyPara([
          bold('Cell Colour (diverging scale). '),
          run('A seven-stop diverging colour scale maps percentage change to colour: deep red (#7f1d1d) at −75%, transitioning through light red at −25%, neutral off-white at 0% (the 2019–20 baseline), light blue at +25%, and deep blue (#1e3a8a) at +35%. This scale makes below-baseline, near-baseline, and above-baseline values simultaneously distinguishable, supporting three-way interpretation across twelve rows and ten columns.'),
        ]),
        bodyPara([
          bold('Cell Annotations. '),
          run('Exact percentage values (e.g., "−53%", "−6%", "+36%") are printed inside each cell in dark ink, enabling precise numerical comparison that the colour encoding alone cannot provide at extreme values.'),
        ]),
        bodyPara([
          bold('Orange Dashed Border. '),
          run('An orange (#f97316) dashed rectangle circumscribes the entire 2020–21 lockdown row, visually isolating the year of peak disruption and drawing the reader's attention to the pandemic event. A labelled annotation "◄ COVID lockdown" is positioned to the right.'),
        ]),
        bodyPara([
          bold('Analytic Annotations. '),
          run('Three labels are algorithmically placed: "▼ largest drop" marks the column with the worst lockdown decline (★#3 Arthropathies); "▲ most resilient" marks the column with the smallest decline in 2020–21 (★#1 Labour & Delivery); and "↑ rebound above baseline" marks the first year and column where admissions exceeded the 2019–20 level after the lockdown (★#6 Cataracts, 2021–22, +36%). These annotations transform descriptive observation into directed analytical interpretation.'),
        ]),

        // ── Unique Observation ──────────────────────────────────
        sectionHead('Unique Observation'),
        bodyPara([
          run('The most important and non-obvious insight yielded by this dual-view design is that '),
          bold('cumulative burden and pandemic resilience are structurally uncorrelated'),
          run(' across the highest-volume diagnostic categories. The treemap (Figure 1) identifies ★#1 Labour & Delivery as the single highest-burden diagnosis over the entire 2012–2023 study period, with 8.77 million admissions — a block so large it rivals entire mid-sized ICD chapters. One might reasonably assume that a category of such clinical dominance would also exhibit robust admission continuity during a public health crisis. The heatmap (Figure 2) refutes this assumption selectively and revealingly. Labour & Delivery recorded a decline of only −6% in 2020–21, appearing in the rightmost column as the most resilient of the ten diagnoses — consistent with the time-critical, non-deferrable nature of obstetric care. In sharp contrast, ★#3 Arthropathies — the third highest-burden diagnosis in the treemap — collapsed by −53% in the same year, appearing in the leftmost heatmap column as the most disrupted. Similarly, ★#6 Cataracts (Lens Disorders) fell −46%, revealing that Eye & Ear, which the treemap shows to be almost entirely composed of this single diagnosis, was among the most acutely disrupted chapters despite its high absolute volume. The explanation is clinically coherent: elective, deferrable procedures (joint replacement, cataract surgery, hernia repair, spinal treatments) were systematically cancelled during the 2020–21 lockdown, whereas emergency and obstetric services could not be postponed. This distinction — between '),
          italic('volume-dominant but deferrable'),
          run(' and '),
          italic('volume-dominant but non-deferrable'),
          run(' care — is entirely invisible in the treemap alone, and cannot be extracted from a standard bar or line chart without disaggregating every diagnosis individually. The heatmap further reveals that recovery after 2020–21 was incomplete and uneven: by 2022–23, ★#6 Cataracts had rebounded to +36% above baseline (a surge driven by the NHS elective recovery programme), while ★#3 Arthropathies remained −17% below, and ★#4 Intestinal Diseases persisted at −8%, indicating that the structural disruption to elective services extended well beyond the single lockdown year. This pattern — partial rebound in some categories, persistent suppression in others — is precisely the type of temporal, multi-diagnosis comparison that a treemap is unable to encode and that a line chart would require ten separate panels to display. The dual-view design surfaces it in a single, cohesive analytical frame.'),
        ]),

        // ── Data Preparation ────────────────────────────────────
        sectionHead('Data Preparation'),
        bodyPara([
          run('The raw ONS hospital admissions dataset was processed in five stages. '),
          bold('(1) Cleaning and standardisation: '),
          run('Column names were normalised (renaming the primary code column, the description column, and the year column to consistent identifiers); numeric fields including Admissions, Emergency, Male, Female, Waiting List, Planned, Mean Age, Mean Length of Stay, and Mean Time Waited were coerced to numeric type with non-numeric entries converted to NaN. Rows with missing financial year or diagnosis description were dropped. The partial 2023–24 year was excluded as it represented an incomplete observation window.'),
        ]),
        bodyPara([
          bold('(2) ICD chapter derivation: '),
          run('A two-character ICD-10 chapter variable was constructed by extracting the leading letter of each diagnosis code and mapping it to one of 18 clinically meaningful chapter labels (e.g., K → Digestive, O → Pregnancy & Childbirth, M → Musculoskeletal). This derived variable provides the outer hierarchical level required for the treemap. Without it, the dataset contains only flat diagnosis codes; the treemap hierarchy would be structurally invalid.'),
        ]),
        bodyPara([
          bold('(3) Aggregation: '),
          run('Admissions were summed across all years (2012–2023) at the chapter–diagnosis level to produce a static measure of cumulative burden for the treemap. Separately, annual admissions were aggregated by diagnosis and financial year for the heatmap computation.'),
        ]),
        bodyPara([
          bold('(4) Top-10 ranking: '),
          run('The ten diagnoses with the highest cumulative admission totals across 2012–2023 were identified and ranked (#1 = highest). These ranks were embedded in the treemap as ★#N labels and gold-bordered cells, and the same ten diagnoses — in identical rank order — were used as the column set for Figure 2, creating the explicit analytical bridge between the two visualisations.'),
        ]),
        bodyPara([
          bold('(5) Baseline normalisation for the heatmap: '),
          run('For each of the ten selected diagnoses, annual admissions from 2012–13 to 2022–23 were expressed as a percentage change relative to the 2019–20 pre-COVID baseline: Δ% = ((admissions_year − admissions_2019-20) / admissions_2019-20) × 100. This normalisation removes the effect of differing absolute scales between diagnoses, enabling direct colour comparison across the matrix. Heatmap columns were ordered by the magnitude of the 2020–21 percentage decline (largest drop leftmost), and values were clipped at −75% / +35% for colour-scale clarity. Short display labels were generated algorithmically by removing stop-words and truncating to 24 characters; a lookup dictionary preserved preferred abbreviations for the most clinically important terms.'),
        ]),

        rule(),

        // ── URLs ────────────────────────────────────────────────
        sectionHead('Optional Video URL'),
        bodyPara([
          run('[Insert YouTube or Vimeo screen-capture link here — recommended ≤ 2 minutes. The video should demonstrate treemap hover interaction, heatmap cell annotation, and the ★#1–#10 label system, ending with a spoken summary of the cross-visual observation.]'),
        ]),

        sectionHead('Optional Source Code URL'),
        bodyPara([
          run('[Insert GitHub / GitLab repository URL here. The repository should contain '),
          italic('hospital_viz_app_10.py'),
          run(', '),
          italic('requirements.txt'),
          run(', and a '),
          italic('README.md'),
          run(' with run instructions. All functions are documented with inline comments following PEP 257 conventions.]'),
        ]),

        spacer(100),

        // Footer
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 60, after: 0 },
          children: [
            new TextRun({ text: 'COMP4037 Research Methods · Coursework 2 · ONS Hospital Admissions 2012–2023 · Python 3 / Pandas / Plotly / Streamlit', font: 'Arial', size: 17, color: C.lightslate })
          ]
        }),

      ]
    }
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('cw2_report.docx', buf);
  console.log('Done: cw2_report.docx');
});
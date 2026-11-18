#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "AgenticAL_Technical_Report_Draft_2.docx"
FIG = ROOT / "figures" / "technical_report"
COLORS = {
    "phlox": "CAA9F3",
    "verbena": "B37AD4",
    "periwinkle": "7997E6",
    "atlantis": "206ABC",
    "phthalo": "0E155E",
    "pale": "F5F2FC",
    "border": "D9D9D9",
    "text": "171A21",
    "muted": "5D6675",
}
TARGETS = ["D2R", "Mpro", "TYK2", "USP7"]
LEADERS = {"D2R": "balanced", "Mpro": "explore-heavy", "TYK2": "exploit", "USP7": "explore-heavy"}
VARIANT_LABELS = {
    "full_reference": "Full information",
    "minus_uncertainty": "No uncertainty",
    "minus_diversity": "No diversity",
    "minus_memory": "No memory",
    "shuffled_evidence": "Shuffled evidence",
}


def set_cell_fill(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), color)


def set_cell_margins(cell, top=110, start=110, bottom=110, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), "5")
        tag.set(qn("w:color"), COLORS["border"])


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def add_field(paragraph, field: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def add_hyperlink(paragraph, text: str, url: str) -> None:
    part = paragraph.part
    rid = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), COLORS["atlantis"])
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.extend([color, underline])
    run.append(r_pr)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_paragraph(doc, text: str = "", style=None, bold_lead: str | None = None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.line_spacing = 1.12
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead) :])
    else:
        p.add_run(text)
    return p


def add_bullets(doc, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.first_line_indent = Inches(-0.14)
        p.add_run(item)


def add_table(doc, headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    repeat_table_header(header)
    prevent_row_split(header)
    for i, value in enumerate(headers):
        cell = header.cells[i]
        set_cell_fill(cell, COLORS["phthalo"])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(value)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9)
        if widths:
            cell.width = Inches(widths[i])
    for row_i, values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        cells = row.cells
        for i, value in enumerate(values):
            cell = cells[i]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_i % 2 == 1:
                set_cell_fill(cell, COLORS["pale"])
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            run.font.size = Pt(8.7)
            if widths:
                cell.width = Inches(widths[i])
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    return table


def add_figure(doc, filename: str, caption: str, width=6.75) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(FIG / filename), width=Inches(width))
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_after = Pt(10)
    cap.add_run(caption)


def section_heading(doc, text: str, level=1) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(COLORS["text"])
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.12

    title = styles["Title"]
    title.font.name = "Aptos Display"
    title._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
    title._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
    title.font.size = Pt(30)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title_ppr = title._element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)

    for name, size in (("Heading 1", 17), ("Heading 2", 13), ("Heading 3", 11)):
        style = styles[name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        style.paragraph_format.space_after = Pt(6)

    caption = styles["Caption"]
    caption.font.name = "Aptos"
    caption.font.size = Pt(8.8)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor.from_string(COLORS["muted"])

    header = section.header.paragraphs[0]
    header.text = "Agentic Active Learning Benchmark"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor.from_string(COLORS["muted"])
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Page ")
    add_field(footer, " PAGE ")
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(COLORS["muted"])


def build() -> None:
    agent = pd.read_csv(ROOT / "results" / "codex_agent_dev" / "trajectories.csv")
    decisions = pd.read_csv(ROOT / "results" / "codex_agent_dev" / "decisions.csv")
    baseline = pd.read_csv(ROOT / "results" / "baselines_10seeds" / "trajectories.csv")
    paired = pd.read_csv(FIG / "paired_top2_summary.csv")
    calibration = pd.read_csv(ROOT / "results" / "calibration" / "legacy_scoring_summary.csv")
    availability = pd.read_csv(
        ROOT / "results" / "availability_constrained" / "analysis" / "constrained_policy_leaders.csv"
    )
    availability_agent = pd.read_csv(
        ROOT / "results" / "availability_constrained" / "agent_full" / "canonical_40_endpoints.csv"
    )
    ablation_path = ROOT / "results" / "information_ablations" / "full" / "endpoint_metrics.csv"
    ablations = pd.read_csv(ablation_path) if ablation_path.exists() else None
    endpoint = agent[agent.cycle == 10]
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.add_run("Agentic Active Learning Benchmark Technical Report")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_before = Pt(6)
    subtitle.paragraph_format.space_after = Pt(18)
    r = subtitle.add_run("GPT 5 6 Sol Strategy Controller Across Four Ligand Discovery Targets")
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor.from_string(COLORS["muted"])
    meta = doc.add_paragraph()
    meta.add_run("Technical Draft 2\n").bold = True
    meta.add_run("11 September 2026\n")
    meta.add_run("Repository  vaidehisaxena AL neurips\n")
    meta.add_run("Branch  codex agentic al")
    meta.paragraph_format.space_after = Pt(28)

    add_paragraph(
        doc,
        "This report evaluates whether a language model can control the balance between exploiting strong predictions, investigating uncertain candidates, and sampling chemically different candidates during an active learning campaign. The main result is straightforward: the controller was consistently much better than random selection, but it did not beat the strongest target-specific fixed strategy on every dataset.",
    )
    add_paragraph(
        doc,
        "The core study completed 40 agent campaigns and 360 fixed-policy campaigns. The next phase added 200 prespecified information-ablation campaigns, 40 availability-constrained agent campaigns, and 360 availability-constrained fixed-policy campaigns. Every campaign began with 60 randomly measured compounds, added ten batches of 30, and finished after 360 measurements.",
    )

    palette = doc.add_table(rows=1, cols=5)
    palette.alignment = WD_TABLE_ALIGNMENT.CENTER
    palette.autofit = False
    for cell, color in zip(palette.rows[0].cells, ["phlox", "verbena", "periwinkle", "atlantis", "phthalo"]):
        set_cell_fill(cell, COLORS[color])
        cell.width = Inches(1.2)
        cell.text = " "
    doc.add_page_break()

    section_heading(doc, "Executive Summary")
    add_paragraph(doc, "The simplest reading of the experiment is:")
    add_bullets(
        doc,
        [
            "The agent reliably learned where useful compounds were. It beat random selection in all 40 matched target and seed comparisons.",
            "The agent was strongest on USP7, recovering 68.3 percent of the truly top 2 percent compounds after measuring 360 candidates.",
            "The agent found the single best compound in every Mpro, TYK2, and USP7 run. It found it in seven of ten D2R runs.",
            "The agent was approximately tied with the strongest fixed policy on Mpro and USP7, but it was worse than the strongest fixed policy on D2R and TYK2.",
            "The acquisition mix genuinely changed with context. TYK2 became strongly exploitative, while Mpro continued to reserve most of each batch for uncertainty and diversity.",
            "The four information ablations changed endpoint recall only slightly. Shuffling explanatory evidence reproduced the full controller path almost everywhere, suggesting that this evidence channel did not drive most allocation choices.",
            "Numerical forecasts were usually successful when testable, but 19.8 percent of the original expected-outcome statements were too qualitative for the threshold scorer.",
            "Restricting post-initial availability to 80 percent reduced agent recall on every target, although the constrained agent still beat the best constrained fixed policy on Mpro and USP7.",
        ],
    )
    add_figure(
        doc,
        "endpoint_top2_comparison.png",
        "Figure 1  Mean top 2 percent recall at the final 360 compound budget. The GPT-5.6 Sol controller is compared with random selection and the strongest fixed policy for each target.",
        width=6.25,
    )

    section_heading(doc, "What Was Tested")
    add_paragraph(
        doc,
        "Active learning tries to discover valuable compounds without measuring the entire candidate library. A Gaussian process model predicts both an expected affinity and an uncertainty for every unmeasured compound. The strategy controller does not calculate affinity and does not directly inspect hidden answers. Instead, it decides how to divide each 30-compound batch among three deterministic queues.",
    )
    add_table(
        doc,
        ["Queue", "Plain language meaning", "Purpose"],
        [
            ["Exploit", "Choose compounds with the highest predicted affinity", "Harvest likely hits"],
            ["Uncertain", "Choose compounds for which the model is least certain", "Learn where the model is weak"],
            ["Diverse", "Choose compounds unlike those already measured", "Broaden chemical coverage"],
        ],
        [1.1, 3.5, 2.0],
    )
    add_paragraph(
        doc,
        "At every cycle, GPT-5.6 Sol produced three nonnegative integer allocations that had to sum to 30, along with a rationale and a prediction that could be checked after the assay. The deterministic environment then selected the actual compounds, revealed only those affinities, updated the model, and generated the next authorized summary.",
    )

    section_heading(doc, "Experimental Design")
    add_table(
        doc,
        ["Element", "Frozen setting"],
        [
            ["Targets", "D2R  Mpro  TYK2  USP7"],
            ["Dataset sizes", "2,502  2,749  9,997  1,799 compounds respectively"],
            ["Molecular representation", "Morgan fingerprint radius 4 with 4,096 bits"],
            ["Predictive model", "Exact Gaussian process with a fixed Tanimoto kernel"],
            ["Starting measurements", "60 randomly selected compounds"],
            ["Acquisition", "10 cycles of 30 compounds"],
            ["Final budget", "360 measured compounds per campaign"],
            ["Replicates", "10 paired random seeds per target and method"],
            ["Agent controller", "GPT-5.6 Sol through Codex subagent tasks"],
            ["Fixed-policy benchmark", "9 policies and 360 complete campaigns"],
        ],
        [2.15, 4.45],
    )
    add_paragraph(
        doc,
        "Pairing is important. Each strategy saw the same starting set for a given target and seed, so performance differences are less likely to be caused by an unusually easy or difficult initialization.",
    )

    section_heading(doc, "How to Read the Metrics")
    add_table(
        doc,
        ["Metric", "Meaning", "Better direction"],
        [
            ["Top 1 percent recall", "Share of the very best 1 percent of the full library that was discovered", "Higher"],
            ["Top 2 percent recall", "Share of the best 2 percent that was discovered  this is the main endpoint", "Higher"],
            ["Top 5 percent recall", "Share of the best 5 percent that was discovered", "Higher"],
            ["Simple regret", "Gap between the best discovered affinity and the best affinity in the full library", "Lower  zero is perfect"],
            ["Unique top 5 percent scaffolds", "Number of distinct chemical cores among discovered high performers", "Higher indicates breadth"],
            ["Learning curve", "How recall changes as more compounds are measured", "Higher and earlier"],
        ],
        [1.55, 3.8, 1.25],
    )
    add_paragraph(
        doc,
        "For example, D2R top 2 percent recall of 0.412 means that the controller recovered 41.2 percent of all compounds belonging to the true top 2 percent, after testing only 360 compounds. It does not mean that 41.2 percent of the tested compounds were successful.",
    )

    section_heading(doc, "Results Across Targets")
    rows = []
    for target in TARGETS:
        group = endpoint[endpoint.target == target]
        rows.append(
            [
                target,
                f"{group.top_1p_recall.mean():.3f}",
                f"{group.top_2p_recall.mean():.3f}",
                f"{group.top_5p_recall.mean():.3f}",
                f"{group.simple_regret.mean():.3f}",
                f"{group.unique_top_5p_scaffolds.mean():.1f}",
            ]
        )
    add_table(
        doc,
        ["Target", "Top 1 recall", "Top 2 recall", "Top 5 recall", "Regret", "Top 5 scaffolds"],
        rows,
        [0.85, 1.05, 1.05, 1.05, 0.85, 1.35],
    )
    section_heading(doc, "D2R", level=2)
    add_paragraph(doc, "D2R was the hardest target for the controller relative to the best fixed policy. Mean top 2 percent recall was 0.412, compared with 0.494 for the balanced fixed rule. The paired difference was minus 0.082, with a 95 percent bootstrap interval from minus 0.150 to minus 0.020. The controller found the absolute best compound in seven of ten campaigns.")
    section_heading(doc, "Mpro", level=2)
    add_paragraph(doc, "Mpro was the clearest competitive result. Mean top 2 percent recall was 0.374 for the controller and 0.370 for the explore-heavy fixed policy. The paired difference was only plus 0.004 and its confidence interval crossed zero, so this should be described as a statistical tie rather than a demonstrated win. All ten campaigns found the global best compound.")
    section_heading(doc, "TYK2", level=2)
    add_paragraph(doc, "TYK2 strongly rewarded exploitation. The controller achieved mean top 2 percent recall of 0.448, substantially above random at 0.036, but below pure exploitation at 0.519. The paired deficit to exploitation was 0.071, with a bootstrap interval from 0.021 to 0.125 below zero. All ten campaigns still found the global best compound.")
    section_heading(doc, "USP7", level=2)
    add_paragraph(doc, "USP7 produced the highest absolute recall. The controller recovered 68.3 percent of the true top 2 percent and 60.4 percent of the true top 5 percent. Its top 2 percent recall was close to the explore-heavy fixed policy at 69.4 percent; the paired difference of minus 0.011 had a confidence interval spanning zero. All ten campaigns found the global optimum.")
    add_figure(
        doc,
        "endpoint_recall_levels.png",
        "Figure 2  Mean endpoint recall at three activity thresholds. Error bars show the standard deviation across ten paired seeds.",
        width=6.8,
    )

    section_heading(doc, "Learning Over the Assay Budget")
    add_paragraph(
        doc,
        "The endpoint alone hides when discoveries occurred. The learning curves show that the controller separated from random early and maintained that advantage. On Mpro and USP7, the controller was ahead of the strongest fixed policy for much of the campaign, even when the two finished near one another. On D2R and TYK2, the fixed strategy built a persistent lead.",
    )
    add_figure(
        doc,
        "top2_learning_curves.png",
        "Figure 3  Top 2 percent recall over the 60 to 360 measurement budget. Lines show the mean across ten seeds and shaded bands show approximately 95 percent confidence intervals based on the standard error.",
        width=6.75,
    )

    section_heading(doc, "How the Agent Changed Strategy")
    add_paragraph(
        doc,
        "The controller did not use one allocation everywhere. Across all targets, it shifted from learning-oriented batches toward exploitation as the campaign matured. The average exploit allocation rose from about 8 compounds in cycle 1 to about 19 in cycle 10, while uncertainty and diversity allocations declined.",
    )
    allocation = decisions.groupby("target")[["n_exploit", "n_uncertain", "n_diverse"]].mean().reindex(TARGETS)
    add_table(
        doc,
        ["Target", "Exploit mean", "Uncertain mean", "Diverse mean", "Interpretation"],
        [
            ["D2R", f"{allocation.loc['D2R','n_exploit']:.1f}", f"{allocation.loc['D2R','n_uncertain']:.1f}", f"{allocation.loc['D2R','n_diverse']:.1f}", "Exploration first  strong late exploitation"],
            ["Mpro", f"{allocation.loc['Mpro','n_exploit']:.1f}", f"{allocation.loc['Mpro','n_uncertain']:.1f}", f"{allocation.loc['Mpro','n_diverse']:.1f}", "Persistent uncertainty and diversity"],
            ["TYK2", f"{allocation.loc['TYK2','n_exploit']:.1f}", f"{allocation.loc['TYK2','n_uncertain']:.1f}", f"{allocation.loc['TYK2','n_diverse']:.1f}", "Strongly exploitative after early cycles"],
            ["USP7", f"{allocation.loc['USP7','n_exploit']:.1f}", f"{allocation.loc['USP7','n_uncertain']:.1f}", f"{allocation.loc['USP7','n_diverse']:.1f}", "Mixed early  exploitative late"],
        ],
        [0.8, 0.9, 1.0, 0.9, 3.0],
    )
    add_figure(
        doc,
        "agent_allocations_by_cycle.png",
        "Figure 4  Mean composition of each 30 compound batch by target and cycle. Areas always sum to 30.",
        width=6.75,
    )
    add_paragraph(
        doc,
        "The allocation behavior is scientifically interesting but should not be mistaken for causal proof. The study shows that the controller changed its decisions with state and target. It does not yet show which information caused each change or whether the written rationale faithfully represents the internal reason for the decision.",
    )

    section_heading(doc, "Comparison With Every Fixed Policy")
    add_figure(
        doc,
        "agent_vs_policy_heatmap.png",
        "Figure 5  Mean difference in top 2 percent recall between the controller and each fixed policy. Positive values favor the controller. The Average column is the unweighted mean across the four targets.",
        width=6.55,
    )
    add_paragraph(
        doc,
        "The heatmap shows why a single average can mislead. The controller comfortably beat weak or mismatched policies, especially random and pure uncertainty. It did not consistently beat policies that happened to match a target well. Target-specific results therefore belong in the main paper, not only in supplementary material.",
    )

    section_heading(doc, "Paired Statistical Interpretation")
    rows = []
    for target in TARGETS:
        random_row = paired[(paired.target == target) & (paired.comparator == "random")].iloc[0]
        leader_row = paired[(paired.target == target) & (paired.comparator == LEADERS[target])].iloc[0]
        rows.append(
            [
                target,
                f"+{random_row.paired_mean_difference:.3f}",
                f"{random_row.ci_low:+.3f} to {random_row.ci_high:+.3f}",
                LEADERS[target],
                f"{leader_row.paired_mean_difference:+.3f}",
                f"{leader_row.ci_low:+.3f} to {leader_row.ci_high:+.3f}",
            ]
        )
    add_table(
        doc,
        ["Target", "Vs random", "95 percent CI", "Strongest fixed", "Vs strongest", "95 percent CI"],
        rows,
        [0.65, 0.8, 1.35, 1.25, 0.95, 1.35],
    )
    add_paragraph(
        doc,
        "Every random comparison has a confidence interval entirely above zero. This is strong evidence that the controller is useful compared with unguided sampling. The strongest-policy comparisons are mixed: D2R and TYK2 intervals are below zero, while Mpro and USP7 intervals include zero. The correct conclusion is therefore competitive adaptation, not universal superiority.",
    )

    section_heading(doc, "Forecast Testability and Outcome Scoring")
    overall = calibration[(calibration.scope == "overall") & (calibration.group == "all")].iloc[0]
    add_paragraph(
        doc,
        f"Each written expected outcome was compared with the immediately following 30-compound batch. Of 400 forecasts, {int(overall.scorable_decisions)} contained at least one explicit threshold that the scorer could test; {int(overall.unscorable_decisions)} ({overall.unscorable_rate:.1%}) were too vague for rule-based scoring. Across {int(overall.scored_atomic_claims)} testable atomic claims, {overall.atomic_success_rate:.1%} succeeded. When a forecast contained several clauses, every clause succeeded in {overall.conjunction_success_rate:.1%} of scorable decisions.",
    )
    add_paragraph(
        doc,
        "In simple terms, the controller was usually right when it made a numerical claim, but it did not always make a claim precise enough to check. This is a testability and hit-rate analysis, not probability calibration: the original forecasts did not include numerical probabilities, so reliability curves or Brier scores cannot be computed retrospectively.",
    )
    add_figure(
        doc,
        "forecast_testability_and_success.png",
        "Figure 6  Fraction of forecasts containing supported testable thresholds and fraction of scorable forecasts for which every stated clause succeeded. USP7 often used qualitative wording, producing a lower testability rate.",
        width=6.65,
    )

    section_heading(doc, "Operational Availability Constraint")
    add_paragraph(
        doc,
        "A second benchmark made only a deterministic 80 percent of each library available after the shared 60-compound initialization. Availability was assigned from canonical isomeric SMILES with a salted SHA-256 rule, without using affinity labels. This preserves paired comparisons while mimicking a campaign in which some candidates cannot be purchased or synthesized.",
    )
    availability = availability.set_index("target").reindex(TARGETS)
    full_agent_means = endpoint.groupby("target").top_2p_recall.mean().reindex(TARGETS)
    constrained_agent_means = availability_agent.groupby("target").top_2p_recall.mean().reindex(TARGETS)
    availability_rows = []
    for target in TARGETS:
        row = availability.loc[target]
        availability_rows.append([
            target,
            f"{full_agent_means.loc[target] + 1e-12:.3f}",
            f"{constrained_agent_means.loc[target] + 1e-12:.3f}",
            f"{(constrained_agent_means.loc[target] - full_agent_means.loc[target]):+.3f}",
            f"{row.constrained_top2_mean:.3f} ({row.policy})",
        ])
    add_table(
        doc,
        ["Target", "Agent full", "Agent 80%", "Agent change", "Best fixed at 80%"],
        availability_rows,
        [0.75, 1.0, 1.0, 1.0, 2.35],
    )
    add_paragraph(
        doc,
        "The agent lost recall on every target under the availability restriction: 0.070 on D2R, 0.072 on Mpro, 0.114 on TYK2, and 0.069 on USP7. Even so, the constrained agent exceeded the best constrained fixed policy on Mpro and USP7, while the strongest fixed rule remained ahead on D2R and TYK2. Operational access therefore changes attainable performance without eliminating the original target-to-target heterogeneity.",
    )
    add_figure(
        doc,
        "availability_agent_comparison.png",
        "Figure 7  Mean endpoint top-2% recall for the agent on the full library, the same controller under 80% availability, and the strongest constrained fixed policy. Each value averages ten paired seeds.",
        width=6.65,
    )

    if ablations is not None and set(TARGETS).issubset(set(ablations.target.unique())):
        section_heading(doc, "Prespecified Information Ablations")
        add_paragraph(
            doc,
            "The frozen controller protocol was rerun after withholding one source of information at a time. Removing uncertainty hides posterior spread; removing diversity hides scaffold and novelty cues; removing memory hides prior decisions and outcomes; shuffled evidence preserves the amount of explanatory fingerprint evidence while breaking its feature identity. All other campaign settings remain paired by target and seed.",
        )
        means = ablations.groupby(["target", "variant"]).top_2p_recall.mean().unstack("variant").reindex(TARGETS)
        rows = []
        for target in TARGETS:
            reference = means.loc[target, "full_reference"]
            rows.append([target] + [f"{means.loc[target, variant]:.3f}" for variant in VARIANT_LABELS] + [
                f"{(means.loc[target].drop('full_reference') - reference).min():+.3f}"
            ])
        add_table(
            doc,
            ["Target", "Full", "No uncertainty", "No diversity", "No memory", "Shuffled", "Largest loss"],
            rows,
            [0.65, 0.68, 0.98, 0.88, 0.78, 0.82, 0.82],
        )
        add_paragraph(
            doc,
            "The effects were small. Removing diversity reduced D2R top-2% recall by 0.016 but produced no loss on TYK2 or USP7 and a 0.002 gain on Mpro. Removing memory changed means by −0.004 to +0.004. Removing uncertainty improved D2R by 0.016 and TYK2 by 0.003, was nearly neutral on Mpro, and reduced USP7 by 0.014. Shuffling explanatory evidence changed the mean by exactly zero on every target.",
        )
        add_paragraph(
            doc,
            "The null shuffled-evidence result is especially clear at the decision level: its allocation path was identical to the full controller for all ten D2R, Mpro, and TYK2 seeds and nine of ten USP7 seeds. The controller therefore behaved as if the shuffled fingerprint identities usually carried little decision weight. The ablation full-reference means should be compared only with their paired ablation arms, not with the earlier core benchmark: Codex task inference is not a frozen deterministic API snapshot, so rerunning the same prompt can produce a different controller path.",
        )
        add_paragraph(
            doc,
            "These are causal protocol comparisons only to the extent that each ablation changes a single authorized information channel. Similar performance after an ablation means the removed signal was not necessary under this controller and benchmark; it does not prove that the signal is useless in other active-learning systems.",
        )
        add_figure(
            doc,
            "information_ablation_endpoints.png",
            "Figure 8  Mean endpoint top-2% recall for the full controller and four prespecified information ablations across ten paired seeds per target.",
            width=6.7,
        )

    section_heading(doc, "Validation and Reproducibility")
    add_bullets(
        doc,
        [
            "Zenodo record 17935028 was downloaded in full. All seven published byte sizes and MD5 checksums matched, and the ZIP integrity test passed.",
            "The four target datasets contained unique nonempty SMILES and finite parseable affinities.",
            "The fixed benchmark contained 360 complete campaigns and 3,960 trajectory rows.",
            "The GPT-5.6 Sol benchmark contained 40 complete campaigns, 440 trajectory rows, and 400 decisions.",
            "There were zero invalid allocation totals, wrong endpoint budgets, or duplicate campaign cycle keys.",
            "All selections were unique within a campaign, and all paired methods used identical initial sets for a target and seed.",
            "The expanded code test suite passed 40 tests. Three dependency warnings did not affect pass status.",
            "Legacy forecast scoring covered all 400 decisions and retained 686 atomic threshold claims for evaluation.",
            "The canonical availability benchmark contained 360 complete fixed-policy campaigns, 3,960 trajectory rows, and no unavailable post-initial selections.",
            "The availability-constrained agent benchmark contained 40 complete campaigns, 400 valid decisions, and 12,000 unique post-initial selections; none violated the availability mask.",
            "The information-ablation benchmark contained 200 complete campaigns and 2,000 valid decisions. Ten legacy D2R gate sessions omitted one redundant top-level digest field; the underlying initial-index digests and paired sets were independently reconstructed and matched.",
        ],
    )
    add_paragraph(
        doc,
        "Hidden-label protection was implemented structurally. The state-summary function accepts observed affinities and pool predictions but has no parameter for hidden pool affinity labels. Only the assay environment can reveal labels, and only for selected indices. The controller decisions were made from the emitted safe summaries and prior revealed outcomes.",
    )

    section_heading(doc, "Limitations")
    add_bullets(
        doc,
        [
            "The controller ran as Codex subagent tasks, not through a frozen API model snapshot. This avoids API cost but weakens exact reproducibility of inference settings and task-runtime behavior.",
            "The GP kernel and molecular representation were fixed for fairness. The conclusions may change under a different surrogate or representation.",
            "The expected-outcome scorer recognizes explicit threshold templates. It cannot reliably interpret every qualitative forecast, and the original controller did not state probabilities needed for formal probabilistic calibration.",
            "These are retrospective virtual screens with known labels, not prospective wet-lab experiments. The availability mask approximates access constraints but does not model synthetic routes, assay failures, cost, or real-time procurement.",
            "A usage-limit interruption occurred during expansion. Every campaign resumed from saved checkpoints, and no incomplete campaign entered the final analysis.",
        ],
    )

    section_heading(doc, "Remaining Reproducibility Step")
    add_paragraph(doc, "The only unexecuted item from the frozen next-phase plan is the optional paid API replication check.")
    add_bullets(
        doc,
        [
            "A fail-closed 20-call runner and frozen configuration are ready for GPT-5.6 Sol through the Responses API.",
            "Execution requires both an OPENAI_API_KEY and the explicit paid-check flag; neither condition was supplied during this study, so no chargeable request was made.",
            "Run this check only if exact API replication, response identifiers, and token accounting are necessary for the final submission.",
        ],
    )

    section_heading(doc, "Conclusion")
    add_paragraph(
        doc,
        "GPT-5.6 Sol successfully controlled a constrained active learning strategy across four chemically different target datasets. In the core benchmark it respected every assay budget and consistently outperformed random sampling, but did not beat every optimized fixed policy. The 200-campaign ablation study found only small effects from removing uncertainty, diversity, or memory and essentially no behavioral effect from shuffled explanatory evidence. The availability study showed that losing access to 20 percent of candidates reduces recall while preserving the original target-specific pattern of wins and losses. The strongest claim is therefore that model-guided strategy control is feasible, auditable, and adaptable to operational constraints—not that every supplied explanation is causally used or that the agent is universally superior.",
    )

    section_heading(doc, "Sources and Project Artifacts")
    p = doc.add_paragraph()
    p.add_run("Dataset and published artifacts  ").bold = True
    add_hyperlink(p, "Zenodo record 17935028", "https://zenodo.org/records/17935028")
    p = doc.add_paragraph()
    p.add_run("Upstream implementation  ").bold = True
    add_hyperlink(p, "meyresearch explainable AL", "https://github.com/meyresearch/explainable_AL")
    p = doc.add_paragraph()
    p.add_run("Agentic benchmark repository  ").bold = True
    add_hyperlink(p, "vaidehisaxena AL neurips", "https://github.com/vaidehisaxena/AL_neurips")
    p = doc.add_paragraph()
    p.add_run("Controller model documentation  ").bold = True
    add_hyperlink(p, "GPT-5.6 Sol model page", "https://developers.openai.com/api/docs/models/gpt-5.6-sol")
    core = doc.core_properties
    core.title = "Agentic Active Learning Benchmark Technical Report"
    core.subject = "GPT-5.6 Sol active learning benchmark across D2R Mpro TYK2 and USP7"
    core.author = "Vaidehi Saxena"
    core.keywords = "active learning ligand discovery Gaussian process GPT-5.6 Sol"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()

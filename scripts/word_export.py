"""검증된 Markdown 보고서를 Word 바이트로 변환한다. 원본 파일은 변경하지 않는다."""
import re
from io import BytesIO

from docx import Document
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def add_inline(paragraph, text):
    for part in re.split(r"(\*\*.*?\*\*|`[^`]+`)", text):
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            paragraph.add_run(part[1:-1])
        else:
            paragraph.add_run(part)


def markdown_to_docx(markdown):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.8)
    section.left_margin = section.right_margin = Cm(1.5)
    for name in ["Normal", "Title", "Heading 1", "Heading 2", "Heading 3", "List Bullet"]:
        style = doc.styles[name]
        style.font.name = "맑은 고딕"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    doc.styles["Normal"].font.size = Pt(10)
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [s.strip().replace(r"\|", "|") for s in re.split(r"(?<!\\)\|", lines[index].strip())[1:-1]]
                if not all(re.fullmatch(r":?-+:?", s.replace(" ", "")) for s in cells):
                    rows.append(cells)
                index += 1
            if not rows:
                continue
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for number, row in enumerate(rows):
                cells = table.add_row().cells
                for cell, value in zip(cells, row):
                    add_inline(cell.paragraphs[0], value)
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(8 if len(row) >= 7 else 9)
                props = table.rows[-1]._tr.get_or_add_trPr()
                props.append(OxmlElement("w:cantSplit"))
                if number == 0:
                    props.append(OxmlElement("w:tblHeader"))
            doc.add_paragraph()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading[1]) - 1
            paragraph = doc.add_heading(level=min(level, 3))
            add_inline(paragraph, heading[2])
            if level == 0:
                doc.core_properties.title = heading[2]
        elif line.startswith("- "):
            add_inline(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            add_inline(doc.add_paragraph(), line)
        index += 1
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run("페이지 ")
    page = OxmlElement("w:fldSimple")
    page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

"""
cleaning.py — PDF extraction, per-source cleaning, and hierarchical chunking.
Run this first. Produces chunks.jsonl, consumed by indexing.py.
"""
import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import fitz
import pdfplumber

# ============================================================
# PER-SOURCE CLEANING CONFIG
# ============================================================
SOURCE_CLEANING_RULES = {
    "Hong Kong Judiciary - Labour Tribunal.pdf": {
        "strip_line_patterns": [
            r'^Hong Kong Judiciary - Labour Tribunal\s*$',
            r'Court Serv\s*\n?\s*&\s*Facilities\s*>\s*Guide to\s*>\s*Labour Tribunal',
            r'^Labour Tribunal PDF version.*$',
            r',?\s*HKT\s*[\d:]+\s*\d+\s*A\s*\.\d+\s*A\s*M\s*A?\s*bps\s*R?',
        ],
        "strip_phrases": [
            "Court Serv", "& Facilities", "e-", "Court Security",
            "Court Diary", "Jury", "Judgments & Legal Reference",
            "Publications", "Press Releases & Other Information",
        ],
    },
}
DEFAULT_RULES = {"strip_line_patterns": [], "strip_phrases": []}


def clean_text(text: str, source_file: str = None) -> str:
    text = re.sub(r'(?i)page\s*\d+\s*(of\s*\d+)?', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M', '', text)
    text = re.sub(r'(?m)^\s*(\d+\s*)+$', '', text)
    text = re.sub(r'[\d\s]{30,}', ' ', text)
    text = re.sub(r'\d+\.\d+\s*Mbps', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'(?i)last revision date\s*:\s*\d{1,2}-\d{1,2}-\d{4}', '', text)
    text = re.sub(r'(?i)copyright\s*[©c]\s*\d{4}', '', text)
    text = re.sub(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*,\s*[A-Za-z]+', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    for phrase in ["This booklet is issued free of charge.",
                   "This is a sample document for reference only."]:
        text = text.replace(phrase, "")
    for phrase in ["Forms > Forms > Visas >", "About Us", "Courts", "Court Services",
                   "e-Courts", "Sitemap", "Important notices", "Privacy policy", "Contact us"]:
        text = text.replace(phrase, "")

    rules = SOURCE_CLEANING_RULES.get(source_file, DEFAULT_RULES)
    for pattern in rules["strip_line_patterns"]:
        text = re.sub(rf'(?m){pattern}', '', text)
    for phrase in rules["strip_phrases"]:
        text = text.replace(phrase, "")

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    return text.strip()


def extract_pdf(pdf_path: str) -> str:
    all_text = []
    fname = os.path.basename(pdf_path)

    if "CoP_Eng" in fname or "Code of Practice" in fname:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            if text.strip():
                cleaned = clean_text(text, source_file=fname)
                if len(cleaned) > 50:
                    all_text.append(cleaned)
        return "\n\n".join(all_text)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            for table in page.extract_tables():
                if table:
                    rows = []
                    for row in table:
                        if any(cell for cell in row if cell and cell.strip()):
                            rows.append(" | ".join(str(c).strip() if c else "" for c in row))
                    if rows:
                        page_text += "\nTABLE:\n" + "\n".join(rows) + "\nEND_TABLE"
            if page_text.strip():
                cleaned = clean_text(page_text, source_file=fname)
                if len(cleaned) > 50:
                    all_text.append(cleaned)
    return "\n\n".join(all_text)


AUTHORITY_MAP = {
    "EO_guide": "statute", "CoP_Eng": "gov_guidance", "FDHguideEnglish": "gov_guidance",
    "Standard Employment Contract": "gov_guidance", "Hong Kong Judiciary": "gov_guidance",
    "ID(E)969": "gov_guidance",
}


def clean_all_pdfs(input_dir: str, output_dir: str = "./cleaned_texts"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        raw_text = extract_pdf(os.path.join(input_dir, fname))
        if not raw_text.strip():
            print(f"WARNING: no text extracted from {fname}")
            continue
        authority = next((v for k, v in AUTHORITY_MAP.items() if k in fname), "unknown")
        header = f"[SOURCE_FILE: {fname}] [AUTHORITY: {authority}]\n\n"
        out_path = os.path.join(output_dir, fname.replace('.pdf', '.txt'))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(header + raw_text)
        print(f"Saved: {out_path}")


# ============================================================
# HIERARCHICAL CHUNKING
# ============================================================
HEADER_PATTERNS = [
    (re.compile(r'^Chapter\s+(\d+)\s*[:\-]\s*(.+)$', re.I), 'chapter', 0),
    (re.compile(r'^Appendix\s+(\d+)\s*[:\-]?\s*(.*)$', re.I), 'appendix', 0),
    (re.compile(r'^([IVXLCDM]+)\.\s+(.+)$'), 'part', 0),
    (re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)$'), 'subsection', 2),
    (re.compile(r'^(\d+\.\d+)\s+(.+)$'), 'section', 1),
    (re.compile(r'^(\d+)\.\s+(.{0,15}[A-Za-z].+)$'), 'paragraph', 1),
    (re.compile(r'^\(([a-z])\)\s+(.+)$'), 'subclause_alpha', 3),
    (re.compile(r'^\(([ivx]+)\)\s+(.+)$', re.I), 'subclause_roman', 4),
]
TABLE_START = re.compile(r'^TABLE:$')
TABLE_END = re.compile(r'^END_TABLE$')
MAX_LEAF_CHARS = 2200


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    source_authority: str
    section_path: str
    section_id: str
    parent_section_id: Optional[str]
    heading_text: str
    level: str
    is_table: bool
    topic_tag: list = field(default_factory=list)
    language: str = "en"
    content: str = ""


TOPIC_KEYWORDS = {
    "rest_days": ["rest day", "rest days"], "annual_leave": ["annual leave"],
    "sickness_allowance": ["sickness allowance", "sick leave"],
    "termination": ["termination", "dismissal", "dismissed"],
    "severance_long_service": ["severance payment", "long service payment", "redundancy"],
    "wages": ["wages", "wage payment", "minimum allowable wage"],
    "recruitment_fees": ["commission", "placement fee", "prescribed commission"],
    "mpf": ["mandatory provident fund", "MPF", "offsetting"],
    "visa_immigration": ["visa", "extension of stay", "Immigration Department"],
    "dispute_resolution": ["Labour Tribunal", "claim", "conciliation"],
    "accommodation": ["accommodation", "live-in", "Schedule of Accommodation"],
}


def tag_topics(text: str) -> list:
    text_lower = text.lower()
    return [tag for tag, kws in TOPIC_KEYWORDS.items() if any(kw.lower() in text_lower for kw in kws)]


def strip_table_of_contents(text: str) -> str:
    lines = text.split("\n")
    toc_start = None
    for i, line in enumerate(lines):
        if re.match(r'^(Table of )?Contents\b', line.strip(), re.I):
            toc_start = i
            break
    if toc_start is None:
        return text

    search_limit = min(toc_start + 200, len(lines) - 1)

    def has_dot_leader(line):
        return bool(re.search(r'\.{4,}', line))

    for j in range(toc_start + 1, search_limit):
        candidate = lines[j].strip()
        next_line = lines[j + 1].strip()
        is_header_like = (re.match(r'^([IVXLCDM]+)\.\s+\w', candidate) or
                           re.match(r'^\d+\.\s+\w', candidate) or
                           re.match(r'^\d+\.\d+\s+\w', candidate))
        next_is_prose = (len(next_line) > 60
                          and not re.match(r'^([IVXLCDM]+|\d+(\.\d+)?)\.?\s', next_line)
                          and not has_dot_leader(candidate)
                          and not has_dot_leader(next_line))
        if is_header_like and next_is_prose:
            return "\n".join(lines[j:])
    return text


def split_if_oversized(text: str) -> list[str]:
    if len(text) <= MAX_LEAF_CHARS:
        return [text]
    paras = text.split("\n\n")
    out, current = [], ""
    for p in paras:
        if len(current) + len(p) > MAX_LEAF_CHARS and current:
            out.append(current.strip())
            current = p
        else:
            current += "\n\n" + p if current else p
    if current.strip():
        out.append(current.strip())
    return out


def parse_hierarchical(raw_text: str, source_file: str, source_authority: str) -> list[Chunk]:
    lines = raw_text.split("\n")
    chunks, stack, buffer = [], [], []
    in_table, table_buffer, chunk_counter = False, [], 0

    def flush_buffer():
        nonlocal buffer, chunk_counter
        text = "\n".join(buffer).strip()
        buffer = []
        if len(text) < 20:
            return
        section_path = " > ".join(s['heading'] for s in stack) if stack else "unstructured"
        current = stack[-1] if stack else None
        for sub_text in split_if_oversized(text):
            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"{source_file}::{chunk_counter}", source_file=source_file,
                source_authority=source_authority, section_path=section_path,
                section_id=current['section_id'] if current else "n/a",
                parent_section_id=stack[-2]['section_id'] if len(stack) >= 2 else None,
                heading_text=current['heading'] if current else "n/a",
                level=current['level_name'] if current else "unstructured",
                is_table=False, topic_tag=tag_topics(sub_text), content=sub_text,
            ))

    def flush_table():
        nonlocal table_buffer, chunk_counter
        if not table_buffer:
            return
        text = "TABLE:\n" + "\n".join(table_buffer) + "\nEND_TABLE"
        table_buffer = []
        section_path = " > ".join(s['heading'] for s in stack) if stack else "unstructured"
        current = stack[-1] if stack else None
        chunk_counter += 1
        chunks.append(Chunk(
            chunk_id=f"{source_file}::{chunk_counter}", source_file=source_file,
            source_authority=source_authority, section_path=section_path,
            section_id=current['section_id'] if current else "n/a",
            parent_section_id=stack[-2]['section_id'] if len(stack) >= 2 else None,
            heading_text=current['heading'] if current else "n/a",
            level=current['level_name'] if current else "unstructured",
            is_table=True, topic_tag=tag_topics(text), content=text,
        ))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue
        if TABLE_START.match(stripped):
            flush_buffer(); in_table = True; continue
        if TABLE_END.match(stripped):
            in_table = False; flush_table(); continue
        if in_table:
            table_buffer.append(stripped); continue

        matched = False
        for pattern, level_name, level_rank in HEADER_PATTERNS:
            m = pattern.match(stripped)
            if m:
                flush_buffer()
                sec_id, heading = m.group(1), m.group(2).strip()
                while stack and stack[-1]['level_rank'] >= level_rank:
                    stack.pop()
                stack.append({'level_rank': level_rank, 'heading': f"{sec_id} {heading}".strip(),
                              'section_id': sec_id, 'level_name': level_name})
                matched = True
                break
        if not matched:
            buffer.append(stripped)

    flush_buffer()
    flush_table()
    return chunks


def build_corpus(input_dir="./cleaned_texts") -> list[Chunk]:
    all_chunks = []
    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(input_dir, fname), encoding="utf-8") as f:
            raw = f.read()
        m = re.match(r'\[SOURCE_FILE:\s*(.+?)\]\s*\[AUTHORITY:\s*(.+?)\]', raw)
        src_file = m.group(1) if m else fname
        authority = m.group(2) if m else "unknown"
        body = raw[m.end():].strip() if m else raw
        body = strip_table_of_contents(body)
        chunks = parse_hierarchical(body, src_file, authority)
        all_chunks.extend(chunks)
        print(f"{fname}: {len(chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    clean_all_pdfs("./pdfs", "./cleaned_texts")
    corpus = build_corpus("./cleaned_texts")
    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"\nTotal chunks: {len(corpus)} → saved to chunks.jsonl")
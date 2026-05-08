import json
import re
from pathlib import Path

import fitz


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PDF_PATH = PROJECT_DIR / "data" / "2023 - Multi-sensor detection of spring breakup phenology of Canada's lakes.pdf"
OUTPUT_DIR = BASE_DIR / "parsed_outputs"
CHUNKS_PATH = OUTPUT_DIR / "day13_chunks.jsonl"
REPORT_PATH = OUTPUT_DIR / "day13_parse_report.md"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MIN_PAGE_TEXT_LENGTH = 80


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping text readable for RAG chunks."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad\n", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end

        if target_end < len(text):
            search_start = start + chunk_size // 2
            sentence_candidates = [
                text.rfind(separator, search_start, target_end)
                for separator in [". ", "? ", "! ", "; "]
            ]
            sentence_end = max(sentence_candidates)

            if sentence_end != -1:
                end = sentence_end + 1
            else:
                word_end = text.rfind(" ", search_start, target_end)
                if word_end != -1:
                    end = word_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(0, end - overlap)
        word_boundary = text.find(" ", next_start, min(next_start + 80, len(text)))
        start = word_boundary + 1 if word_boundary != -1 else next_start
    return chunks


def parse_pdf() -> tuple[list[dict], dict]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    chunks = []
    empty_pages = []
    parsed_pages = 0

    for page_index, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        if len(text) < MIN_PAGE_TEXT_LENGTH:
            empty_pages.append(page_index)
            continue

        parsed_pages += 1
        page_chunks = split_text(text)

        for local_chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = f"day13_p{page_index:03d}_c{local_chunk_index:02d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": PDF_PATH.name,
                    "page": page_index,
                    "text": chunk_text,
                    "text_length": len(chunk_text),
                }
            )

    report = {
        "pdf_path": str(PDF_PATH),
        "total_pages": doc.page_count,
        "parsed_pages": parsed_pages,
        "empty_or_short_pages": empty_pages,
        "chunk_count": len(chunks),
        "min_chunk_length": min((item["text_length"] for item in chunks), default=0),
        "max_chunk_length": max((item["text_length"] for item in chunks), default=0),
        "avg_chunk_length": round(
            sum(item["text_length"] for item in chunks) / len(chunks),
            2,
        )
        if chunks
        else 0,
    }
    doc.close()
    return chunks, report


def write_chunks(chunks: list[dict]) -> None:
    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_report(report: dict, chunks: list[dict]) -> None:
    sample_chunks = chunks[:3]
    lines = [
        "# Day13 PDF 解析质量报告",
        "",
        "## 1. 基本信息",
        "",
        f"- PDF 文件: `{report['pdf_path']}`",
        f"- PDF 总页数: {report['total_pages']}",
        f"- 成功解析页数: {report['parsed_pages']}",
        f"- 空页或过短页: {report['empty_or_short_pages']}",
        f"- Chunk 数量: {report['chunk_count']}",
        f"- 最短 Chunk 长度: {report['min_chunk_length']}",
        f"- 最长 Chunk 长度: {report['max_chunk_length']}",
        f"- 平均 Chunk 长度: {report['avg_chunk_length']}",
        "",
        "## 2. Chunk 元数据结构",
        "",
        "每条 JSONL 记录包含：",
        "",
        "```text",
        "chunk_id, source, page, text, text_length",
        "```",
        "",
        "## 3. 样例 Chunk",
        "",
    ]

    for chunk in sample_chunks:
        preview = chunk["text"][:500].replace("\n", " ")
        lines.extend(
            [
                f"### {chunk['chunk_id']}",
                "",
                f"- page: {chunk['page']}",
                f"- text_length: {chunk['text_length']}",
                "",
                "```text",
                preview,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 下一步检查建议",
            "",
            "- 人工抽查样例 chunk 是否存在明显乱码。",
            "- 观察页眉、页脚、参考文献编号是否造成噪声。",
            "- Day14 可将 `day13_chunks.jsonl` 接入 Milvus 入库流程。",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    chunks, report = parse_pdf()
    write_chunks(chunks)
    write_report(report, chunks)

    print("Day13 PDF 解析完成")
    print(f"PDF 总页数: {report['total_pages']}")
    print(f"成功解析页数: {report['parsed_pages']}")
    print(f"Chunk 数量: {report['chunk_count']}")
    print(f"JSONL 输出: {CHUNKS_PATH}")
    print(f"报告输出: {REPORT_PATH}")


if __name__ == "__main__":
    main()

import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PDF_PATH = PROJECT_DIR / "data" / "2023 - Multi-sensor detection of spring breakup phenology of Canada's lakes.pdf"
OUTPUT_DIR = BASE_DIR / "parsed_outputs"
ELEMENTS_PATH = OUTPUT_DIR / "day13_unstructured_elements.jsonl"
REPORT_PATH = OUTPUT_DIR / "day13_unstructured_report.md"


def require_unstructured():
    try:
        from unstructured.partition.pdf import partition_pdf
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        if missing_module == "unstructured_inference":
            raise ModuleNotFoundError(
                "缺少 unstructured_inference。请先运行: pip install unstructured-inference"
            ) from exc

        raise ModuleNotFoundError(
            "缺少 unstructured。请先运行: pip install \"unstructured[pdf]\""
        ) from exc

    return partition_pdf


def element_to_record(element, index: int) -> dict:
    metadata = element.metadata.to_dict() if element.metadata else {}
    text = str(element).strip()

    return {
        "element_id": f"day13_e{index:04d}",
        "type": element.category,
        "source": PDF_PATH.name,
        "page": metadata.get("page_number"),
        "text": text,
        "text_length": len(text),
        "metadata": metadata,
    }


def parse_pdf_with_unstructured() -> list[dict]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    partition_pdf = require_unstructured()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(PDF_PATH),
        strategy="fast",
        infer_table_structure=False,
    )

    records = []
    for index, element in enumerate(elements, start=1):
        record = element_to_record(element, index)
        if record["text"]:
            records.append(record)

    return records


def write_elements(records: list[dict]) -> None:
    with ELEMENTS_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(records: list[dict]) -> None:
    type_counter = Counter(record["type"] for record in records)
    page_counter = Counter(record["page"] for record in records if record["page"] is not None)
    lengths = [record["text_length"] for record in records]

    lines = [
        "# Day13 Unstructured 元素级解析报告",
        "",
        "## 1. 基本信息",
        "",
        f"- PDF 文件: `{PDF_PATH}`",
        f"- 元素数量: {len(records)}",
        f"- 覆盖页数: {len(page_counter)}",
        f"- 最短元素长度: {min(lengths, default=0)}",
        f"- 最长元素长度: {max(lengths, default=0)}",
        f"- 平均元素长度: {round(sum(lengths) / len(lengths), 2) if lengths else 0}",
        "",
        "## 2. 元素类型统计",
        "",
        "| 元素类型 | 数量 |",
        "| --- | ---: |",
    ]

    for element_type, count in type_counter.most_common():
        lines.append(f"| {element_type} | {count} |")

    lines.extend(
        [
            "",
            "## 3. 样例元素",
            "",
        ]
    )

    for record in records[:10]:
        preview = record["text"][:500].replace("\n", " ")
        lines.extend(
            [
                f"### {record['element_id']}",
                "",
                f"- type: {record['type']}",
                f"- page: {record['page']}",
                f"- text_length: {record['text_length']}",
                "",
                "```text",
                preview,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 与 PyMuPDF 基线的区别",
            "",
            "- PyMuPDF 输出的是页面级纯文本，适合快速、稳定地建立 baseline。",
            "- Unstructured 输出的是元素级结果，例如 Title、NarrativeText、ListItem、Table 等，更适合保留文档结构。",
            "- 本脚本使用 `strategy=\"fast\"`，优先保证本地可运行；如果要识别更复杂表格和图片，可再尝试 `strategy=\"hi_res\"`，但依赖会明显更重。",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = parse_pdf_with_unstructured()
    write_elements(records)
    write_report(records)

    type_counter = Counter(record["type"] for record in records)

    print("Day13 Unstructured 解析完成")
    print(f"元素数量: {len(records)}")
    print("元素类型统计:")
    for element_type, count in type_counter.most_common():
        print(f"- {element_type}: {count}")
    print(f"JSONL 输出: {ELEMENTS_PATH}")
    print(f"报告输出: {REPORT_PATH}")


if __name__ == "__main__":
    main()

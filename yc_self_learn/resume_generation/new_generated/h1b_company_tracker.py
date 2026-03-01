import argparse
from pathlib import Path
from datetime import datetime


def load_companies(excel_path: Path, sheet: str | int | None, company_col: str | None):
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise SystemExit(
            "需要先安装 pandas 才能读取 Excel 文件，请先运行：\n"
            "  pip install pandas openpyxl\n"
        )

    if sheet is None:
        sheet_to_use: str | int = 0
    else:
        # 如果是纯数字字符串，就转成下标；否则当作 sheet 名
        if isinstance(sheet, str) and sheet.isdigit():
            sheet_to_use = int(sheet)
        else:
            sheet_to_use = sheet

    df = pd.read_excel(excel_path, sheet_name=sheet_to_use)

    if df.empty:
        raise SystemExit("Excel 工作表是空的。")

    if company_col is None:
        company_col = str(df.columns[0])

    if company_col not in df.columns:
        raise SystemExit(
            f"找不到公司列 '{company_col}'。可用列名有：{list(df.columns)}"
        )

    series = df[company_col].dropna().astype(str).str.strip()
    companies = [name for name in series.tolist() if name]
    if not companies:
        raise SystemExit(f"列 '{company_col}' 中没有有效的公司名称。")

    return companies, company_col


def write_tracker_md(companies: list[str], output_path: Path, source_excel: Path, company_col: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append("# H1B 大公司投递记录\n")
    lines.append(f"- 数据来源: `{source_excel.name}`（列: `{company_col}`）")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 公司总数: {len(companies)}\n")

    lines.append("> 用法：默认所有公司 `投递状态 = 待投递`，你可以根据实际情况改成 `已投递 / 不考虑 / 面试中` 等。\n")

    for idx, name in enumerate(companies, start=1):
        lines.append(f"### {idx}. {name}")
        lines.append("- **投递状态**: 待投递")
        lines.append("- **最近更新**: ")
        lines.append("- **备注**: \n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="从 H1B 大公司 Excel 列表生成投递记录 Markdown。"
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Excel 文件路径（包含所有支持 H1B 的大公司列表）",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="要读取的工作表名或下标（默认读取第一个 sheet）。例如：Sheet1 或 0。",
    )
    parser.add_argument(
        "--company-col",
        default=None,
        help="公司名称所在的列名（默认使用第一个列）。",
    )
    parser.add_argument(
        "--output",
        default="H1B_Companies_Application_Tracker.md",
        help="输出的 Markdown 文件名（默认: H1B_Companies_Application_Tracker.md）。",
    )

    args = parser.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    if not excel_path.exists():
        raise SystemExit(f"找不到 Excel 文件: {excel_path}")

    companies, company_col = load_companies(
        excel_path=excel_path,
        sheet=args.sheet,
        company_col=args.company_col,
    )

    output_path = Path(args.output).expanduser().resolve()
    write_tracker_md(companies, output_path, excel_path, company_col)

    print(f"已生成投递记录文件: {output_path}")


if __name__ == "__main__":
    main()


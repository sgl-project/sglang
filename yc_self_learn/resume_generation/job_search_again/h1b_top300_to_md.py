"""
从 h1b_top300.xlsx 读取公司列表，取前 N 条生成投递记录 Markdown，输出到 job_search_again 文件夹。
"""
import argparse
from pathlib import Path
from datetime import datetime


def load_companies(
    excel_path: Path,
    sheet: str | int | None,
    company_col: str | None,
    top: int,
    start: int,
):
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

    # 取 [start-1 : start-1+top]，即第 start 到 start+top-1 条（1-based）
    i0 = max(0, start - 1)
    i1 = min(len(df), i0 + top)
    df_slice = df.iloc[i0:i1]

    series = df_slice[company_col].dropna().astype(str).str.strip()
    companies = [name for name in series.tolist() if name]

    # 保留整行数据，方便写出更多列
    extra_cols = [c for c in df_slice.columns if c != company_col]
    rows = []
    for i, name in enumerate(companies):
        row = {"name": name, "rank": start + i}
        for c in extra_cols:
            val = df_slice.iloc[i][c]
            if pd.isna(val):
                row[c] = ""
            else:
                row[c] = str(val).strip()
        rows.append(row)

    return rows, company_col, extra_cols


def load_blacklist(path: Path) -> set[str]:
    """
    从文本文件加载中介 / 外包公司黑名单，每行一个公司名（不区分大小写）。
    文件不存在或为空时返回空集合。
    """
    if not path.exists():
        return set()

    text = path.read_text(encoding="utf-8")
    names: set[str] = set()
    for line in text.splitlines():
        name = line.strip()
        if not name:
            continue
        names.add(name.upper())
    return names


def filter_agencies(rows: list[dict], blacklist: set[str]) -> list[dict]:
    """
    根据黑名单过滤掉中介 / 外包公司。
    保留原始 rank，不重新编号，方便看到全局排名。
    """
    if not blacklist:
        return rows

    filtered: list[dict] = []
    for r in rows:
        name_upper = str(r.get("name", "")).strip().upper()
        if name_upper in blacklist:
            continue
        filtered.append(r)
    return filtered


def categorize_company(name: str) -> str:
    """
    根据公司名称粗略分类到：
    - 科技公司
    - 金融公司
    - 硬件公司
    - 医疗公司
    - 其他
    """
    n = str(name).upper()

    finance_keywords = [
        "BANK",
        "FINANCIAL",
        "CAPITAL ONE",
        "FIDELITY",
        "SACHS",
        "MORGAN",
        "VANGUARD",
        "BLACKROCK",
        "STATE STREET",
        "MASTERCARD",
        "VISA",
        "PAYPAL",
        "GEICO",
        "TRUIST",
        "EQUIFAX",
        "SECURITIES",
        "INSURANCE",
        "METLIFE",
        "DISCOVER",
        "COINBASE",
        "SOCIAL FINANCE",
        "FISERV",
        "FIS ",
        "USAA",
    ]
    health_keywords = [
        "HOSPITAL",
        "CLINIC",
        "MEDICAL",
        "MEDICINE",
        "HEALTH ",
        "PHARMACY",
        "PHARMACEUTICAL",
        "AMGEN",
        "LILLY",
        "BRISTOL-MYERS",
        "THERMO FISHER",
        "ABBVIE",
        "GILEAD",
        "HUMANA",
        "OHIOHEALTH",
        "CHILDRENS",
        "CANCER CENTER",
        "LABORATORIES",
        "MEDTRONIC",
        "BECTON DICKINSON",
        "NIH ",
        "NATIONAL INSTITUTES OF HEALTH",
    ]
    hardware_keywords = [
        "SEMICONDUCTOR",
        "MICRON",
        "INTEL",
        "ADVANCED MICRO DEVICES",
        "QUALCOMM",
        "NVIDIA",
        "KLA ",
        "LAM RESEARCH",
        "ASML",
        "NXP ",
        "XILINX",
        "MARVELL",
        "AUSTIN SEMICONDUCTOR",
        "SAMSUNG",
        "HONEYWELL",
        "DEERE AND COMPANY",
        "CATERPILLAR",
        "MOTOR COMPANY",
        "AUTOMOTIVE",
        "CUMMINS",
    ]
    other_keywords = [
        "WAL-MART",
        "LOWES",
        "HOME DEPOT",
        "SAFEWAY",
        "TARGET",
        "NORDSTROM",
        "WAYFAIR",
        "STARBUCKS",
        "AIRLINES",
        "AIR LINES",
        "DALLAS INDEPENDENT SCHOOL DISTRICT",
        "AECOM ",
        "WSP USA",
        "SLALOM",
        "ZS ASSOCIATES",
        "MCKINSEY",
        "BOSTON CONSULTING GROUP",
        "DELOITTE",
        "ERNST AND YOUNG",
        "KPMG",
        "PWC ",
    ]

    if any(k in n for k in health_keywords):
        return "医疗公司"
    if any(k in n for k in finance_keywords):
        return "金融公司"
    if any(k in n for k in hardware_keywords):
        return "硬件公司"
    if any(k in n for k in other_keywords):
        return "其他"
    return "科技公司"


def is_university_like(name: str) -> bool:
    """
    简单判断是否是学校 / 大学类雇主。
    通过名字中的关键词来粗略识别，比如 UNIVERSITY / UNIV / COLLEGE / INSTITUTE OF TECHNOLOGY 等。
    """
    n = str(name).upper()
    if "UNIVERSITY" in n:
        return True
    if " UNIV " in f" {n} ":
        return True
    if "COLLEGE" in n:
        return True
    if " INSTITUTE OF TECHNOLOGY" in n:
        return True
    if "SCHOOL OF MEDICINE" in n:
        return True
    return False


def filter_universities(rows: list[dict]) -> list[dict]:
    """过滤掉大学 / 学校类雇主。"""
    return [r for r in rows if not is_university_like(r.get("name", ""))]


def renumber_rows(rows: list[dict]) -> list[dict]:
    """
    在过滤之后从 1 开始重新连续编号 rank。
    不修改其它字段。
    """
    new_rows: list[dict] = []
    for i, r in enumerate(rows, start=1):
        r2 = dict(r)
        r2["rank"] = i
        new_rows.append(r2)
    return new_rows


def write_tracker_md(
    rows: list[dict],
    output_path: Path,
    source_excel: Path,
    company_col: str,
    extra_cols: list[str],
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    if rows:
        start_rank = rows[0]["rank"]
        end_rank = rows[-1]["rank"]
        lines.append(f"# H1B Top 公司投递记录（{start_rank}–{end_rank}）\n")
    else:
        lines.append("# H1B Top 公司投递记录\n")
    lines.append(f"- 数据来源: `{source_excel.name}`（公司列: `{company_col}`）")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 本页数量: {len(rows)}\n")
    lines.append("> 用法：默认所有公司 `投递状态 = 待投递`，可改为 `已投递 / 不考虑 / 面试中` 等。\n")
    lines.append("---\n")

    for r in rows:
        idx = r["rank"]
        name = r["name"]
        lines.append(f"### {idx}. {name}")
        lines.append("- **投递状态**: 待投递")
        for c in extra_cols:
            if r.get(c):
                lines.append(f"- **{c}**: {r[c]}")
        lines.append("- **最近更新**: ")
        lines.append("- **备注**: \n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_status_md(
    rows: list[dict],
    output_path: Path,
    source_excel: Path,
    company_col: str,
    with_category: bool = False,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    if rows:
        start_rank = rows[0]["rank"]
        end_rank = rows[-1]["rank"]
        lines.append(f"# H1B Top 公司投递状态码汇总（{start_rank}–{end_rank}）\n")
    else:
        lines.append("# H1B Top 公司投递状态码汇总\n")

    lines.append(f"- 数据来源: `{source_excel.name}`（公司列: `{company_col}`）")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 公司数量: {len(rows)}\n")

    lines.append("## 投递状态码说明")
    lines.append("- `0` = 待投递")
    lines.append("- `1` = 已投递")
    lines.append("- `2` = 面试中")
    lines.append("- `3` = 不考虑\n")
    lines.append("---\n")

    for r in rows:
        idx = r["rank"]
        name = r["name"]
        lines.append(f"### {idx}. {name}")
        if with_category:
            category = categorize_company(name)
            lines.append(f"- **公司类别**: {category}")
        lines.append("- **投递状态码**: 0\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_status_split_by_category(
    rows: list[dict],
    output_dir: Path,
    source_excel: Path,
    company_col: str,
):
    """
    按公司类别拆分为多个 md 文件：
    - 科技公司
    - 金融公司
    - 硬件公司
    - 医疗公司
    - 其他
    """
    buckets: dict[str, list[dict]] = {
        "科技公司": [],
        "金融公司": [],
        "硬件公司": [],
        "医疗公司": [],
        "其他": [],
    }

    for r in rows:
        name = r.get("name", "")
        cat = categorize_company(name)
        if cat not in buckets:
            cat = "其他"
        buckets[cat].append(r)

    mapping = {
        "科技公司": "H1B_Top300_Status_Tech.md",
        "金融公司": "H1B_Top300_Status_Finance.md",
        "硬件公司": "H1B_Top300_Status_Hardware.md",
        "医疗公司": "H1B_Top300_Status_Healthcare.md",
        "其他": "H1B_Top300_Status_Others.md",
    }

    for cat, cat_rows in buckets.items():
        if not cat_rows:
            continue
        filename = mapping[cat]
        out_path = output_dir / filename
        write_status_md(
            cat_rows,
            out_path,
            source_excel,
            company_col,
            with_category=True,
        )


def main():
    base = Path(__file__).resolve().parent
    out_dir = base / "job_search_again"
    # Excel 文件统一放在上一级目录的 new_generated 里
    default_excel = base.parent / "new_generated" / "h1b_top300_2025_2026_merged.xlsx"

    parser = argparse.ArgumentParser(
        description="从 h1b_top300.xlsx 取前 N 条生成投递记录 Markdown，输出到 job_search_again。"
    )
    parser.add_argument(
        "--excel",
        default=str(default_excel),
        help=f"Excel 文件路径（默认: {default_excel.name}）",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="工作表名或下标（默认第一个 sheet）",
    )
    parser.add_argument(
        "--company-col",
        default=None,
        help="公司名称列名（默认第一列）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="取前几条（默认 30）",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="从第几条开始（1-based，默认 1）",
    )
    parser.add_argument(
        "--exclude-agency",
        action="store_true",
        help="排除 agency_blacklist.txt / --blacklist-file 中列出的中介 / 外包公司",
    )
    parser.add_argument(
        "--blacklist-file",
        default=None,
        help="自定义中介 / 外包公司名单文件路径（每行一个公司名）",
    )
    parser.add_argument(
        "--exclude-university",
        action="store_true",
        help="过滤掉大学 / 学校类雇主（UNIVERSITY / UNIV / COLLEGE / INSTITUTE OF TECHNOLOGY 等）",
    )
    parser.add_argument(
        "--renumber",
        action="store_true",
        help="在过滤之后重新从 1 开始连续编号 rank",
    )
    parser.add_argument(
        "--mode",
        choices=["tracker", "status"],
        default="tracker",
        help="输出模式：tracker=完整投递记录，status=仅公司名+投递状态码（默认 tracker）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "输出 md 文件名（默认: "
            "tracker 模式为 job_search_again/H1B_Top<start>_<start+top-1>.md；"
            "status 模式为 job_search_again/H1B_Status_Top<start>_<start+top-1>.md）"
        ),
    )
    parser.add_argument(
        "--with-category",
        action="store_true",
        help="在 status 模式下为每家公司增加“公司类别”字段（科技公司/金融公司/硬件公司/医疗公司/其他）",
    )
    parser.add_argument(
        "--split-by-category",
        action="store_true",
        help="在 status 模式下按公司类别拆分为多个 md 文件（科技/金融/硬件/医疗/其他）",
    )

    args = parser.parse_args()

    excel_path = Path(args.excel).expanduser()
    if not excel_path.is_absolute():
        excel_path = (base / excel_path).resolve()
    else:
        excel_path = excel_path.resolve()
    if not excel_path.exists():
        raise SystemExit(f"找不到 Excel 文件: {excel_path}")

    rows, company_col, extra_cols = load_companies(
        excel_path=excel_path,
        sheet=args.sheet,
        company_col=args.company_col,
        top=args.top,
        start=args.start,
    )

    # 按需过滤中介 / 外包公司（例如 TCS / Infosys / Cognizant / Wipro 等）
    if args.blacklist_file:
        blacklist_path = Path(args.blacklist_file).expanduser()
        if not blacklist_path.is_absolute():
            blacklist_path = (base / blacklist_path).resolve()
    else:
        blacklist_path = base / "agency_blacklist.txt"

    if args.exclude_agency:
        blacklist = load_blacklist(blacklist_path)
        if not blacklist:
            print(
                f"警告: 启用了 --exclude-agency 但黑名单文件为空或不存在: {blacklist_path}"
            )
        rows = filter_agencies(rows, blacklist)

    # 过滤掉大学 / 学校类雇主
    if args.exclude_university:
        rows = filter_universities(rows)

    # 如需，将过滤后的 rank 重新从 1 连续编号
    if args.renumber:
        rows = renumber_rows(rows)

    # 如果在 status 模式下要求按类别拆分，则在这里一次性写出多个文件并返回
    if args.mode == "status" and args.split_by_category:
        if args.output:
            out_base = Path(args.output).expanduser().resolve()
            split_dir = out_base.parent
        else:
            split_dir = out_dir
        write_status_split_by_category(rows, split_dir, excel_path, company_col)
        print(
            f"Done: split {len(rows)} companies by category into multiple files under {split_dir}"
        )
        return

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        start_rank = rows[0]["rank"] if rows else args.start
        end_rank = rows[-1]["rank"] if rows else (args.start + args.top - 1)
        if args.mode == "tracker":
            filename = f"H1B_Top{start_rank}_{end_rank}.md"
        else:
            filename = f"H1B_Status_Top{start_rank}_{end_rank}.md"
        output_path = out_dir / filename

    if args.mode == "tracker":
        write_tracker_md(rows, output_path, excel_path, company_col, extra_cols)
    else:
        write_status_md(
            rows,
            output_path,
            excel_path,
            company_col,
            with_category=args.with_category,
        )

    print(f"Done: {output_path} ({len(rows)} companies, mode={args.mode})")


if __name__ == "__main__":
    main()

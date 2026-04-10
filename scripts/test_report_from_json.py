"""
    Script to generate pdf report from json outputs.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, List, Tuple


def add_src_to_path(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def collect_json_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        return [input_path]

    if input_path.is_dir():
        return sorted([p for p in input_path.glob("*.json") if p.is_file()])

    return []


def generate_reports(json_files: List[Path], output_dir: Path) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    report_module = importlib.import_module("report_gen")
    report_generator_cls: Any = getattr(report_module, "ClinicalReportGenerator")

    output_dir.mkdir(parents=True, exist_ok=True)
    successes: List[Path] = []
    failures: List[Tuple[Path, str]] = []

    for json_file in json_files:
        out_pdf = output_dir / f"{json_file.stem}_test_report.pdf"
        try:
            generator = report_generator_cls(str(json_file))
            generator.generate_pdf(str(out_pdf))
            successes.append(out_pdf)
            print(f"[OK] {json_file} -> {out_pdf}")
        except Exception as exc:
            failures.append((json_file, str(exc)))
            print(f"[FAIL] {json_file} -> {exc}")

    return successes, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PDF report(s) from one JSON report file or all JSON files in a directory. "
            "Useful for quickly validating what works and what fails in report rendering."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/reports/json/raw_report.json",
        help="Path to JSON report file or directory containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports/pdf/test_runs",
        help="Directory where generated test PDFs will be written.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    add_src_to_path(repo_root)

    input_path = (repo_root / args.input_path).resolve() if not Path(args.input_path).is_absolute() else Path(args.input_path)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    json_files = collect_json_files(input_path)
    if not json_files:
        print(f"No JSON files found at: {input_path}")
        return 1

    print(f"Found {len(json_files)} JSON file(s). Starting test report generation...")
    successes, failures = generate_reports(json_files, output_dir)

    print("\n=== Test Report Generation Summary ===")
    print(f"Successful: {len(successes)}")
    print(f"Failed: {len(failures)}")

    if successes:
        print("Generated PDFs:")
        for path in successes:
            print(f" - {path}")

    if failures:
        print("Failed Inputs:")
        for in_path, err in failures:
            print(f" - {in_path}: {err}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALUE_PATTERN = re.compile(r"^\d+(\.\d+)?$")


@dataclass
class LabelIssue:
    line_number: int
    image_name: str
    raw_value: str
    issue: str


def normalize_value(value: str, decimals: int | None = None) -> str:
    value = value.strip()
    value = value.replace(",", ".")
    value = value.replace(" ", "")

    if decimals is not None and VALUE_PATTERN.fullmatch(value):
        number = float(value)
        return f"{number:.{decimals}f}"

    return value


def validate_image_name(image_name: str) -> str | None:
    if image_name != image_name.strip():
        return "image name has leading/trailing spaces"

    if " " in image_name:
        return "image name contains spaces"

    suffix = Path(image_name).suffix.lower()
    if suffix not in VALID_IMAGE_EXTENSIONS:
        return f"invalid image extension: {suffix}"

    return None


def validate_value(value: str) -> str | None:
    if value != value.strip():
        return "value has leading/trailing spaces"

    if " " in value:
        return "value contains spaces"

    if "," in value:
        return "value contains comma instead of dot"

    if not VALUE_PATTERN.fullmatch(value):
        return "value must contain only digits and optional one dot"

    return None


def read_labels(labels_path: Path) -> list[tuple[int, str, str]]:
    rows = []

    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for line_number, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue

            if len(row) != 2:
                rows.append((line_number, "", ""))
                continue

            image_name, value = row
            rows.append((line_number, image_name, value))

    return rows


def validate_and_normalize_labels(
    labels_path: Path,
    images_dir: Path,
    output_path: Path,
    report_path: Path,
    decimals: int | None = None,
    fail_on_missing_images: bool = True,
    fail_on_extra_images: bool = False,
) -> None:
    rows = read_labels(labels_path)

    normalized_rows: list[tuple[str, str]] = []
    issues: list[LabelIssue] = []

    seen_images: set[str] = set()

    for line_number, image_name, raw_value in rows:
        if image_name == "" and raw_value == "":
            issues.append(
                LabelIssue(
                    line_number=line_number,
                    image_name="",
                    raw_value="",
                    issue="invalid row format: expected exactly 2 columns",
                )
            )
            continue

        normalized_image_name = image_name.strip()
        normalized_value = normalize_value(raw_value, decimals=decimals)

        image_issue = validate_image_name(normalized_image_name)
        if image_issue is not None:
            issues.append(
                LabelIssue(
                    line_number=line_number,
                    image_name=image_name,
                    raw_value=raw_value,
                    issue=image_issue,
                )
            )

        value_issue = validate_value(normalized_value)
        if value_issue is not None:
            issues.append(
                LabelIssue(
                    line_number=line_number,
                    image_name=image_name,
                    raw_value=raw_value,
                    issue=value_issue,
                )
            )

        if normalized_image_name in seen_images:
            issues.append(
                LabelIssue(
                    line_number=line_number,
                    image_name=normalized_image_name,
                    raw_value=raw_value,
                    issue="duplicate image in labels file",
                )
            )

        seen_images.add(normalized_image_name)

        image_path = images_dir / normalized_image_name
        if not image_path.exists():
            issues.append(
                LabelIssue(
                    line_number=line_number,
                    image_name=normalized_image_name,
                    raw_value=raw_value,
                    issue="image listed in labels but missing in images directory",
                )
            )

        normalized_rows.append((normalized_image_name, normalized_value))

    images_in_dir = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }

    extra_images = sorted(images_in_dir - seen_images)
    missing_images = sorted(seen_images - images_in_dir)

    if fail_on_extra_images:
        for image_name in extra_images:
            issues.append(
                LabelIssue(
                    line_number=0,
                    image_name=image_name,
                    raw_value="",
                    issue="image exists in images directory but is missing from labels file",
                )
            )

    write_labels(output_path, normalized_rows)
    write_report(report_path, issues, missing_images, extra_images)

    has_blocking_issues = bool(issues)

    if has_blocking_issues:
        print(f"Finished with {len(issues)} issue(s).")
        print(f"Normalized labels saved to: {output_path}")
        print(f"Report saved to: {report_path}")

        if fail_on_missing_images and missing_images:
            raise SystemExit(1)

        if fail_on_extra_images and extra_images:
            raise SystemExit(1)

        raise SystemExit(1)

    print("Labels are valid.")
    print(f"Normalized labels saved to: {output_path}")
    print(f"Report saved to: {report_path}")


def write_labels(output_path: Path, rows: list[tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def write_report(
    report_path: Path,
    issues: list[LabelIssue],
    missing_images: list[str],
    extra_images: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Labels validation report\n\n")

        file.write(f"Total issues: {len(issues)}\n")
        file.write(f"Missing images: {len(missing_images)}\n")
        file.write(f"Extra images: {len(extra_images)}\n\n")

        if issues:
            file.write("## Issues\n\n")
            for issue in issues:
                file.write(
                    f"- line={issue.line_number}, "
                    f"image={issue.image_name!r}, "
                    f"value={issue.raw_value!r}, "
                    f"issue={issue.issue}\n"
                )

        if missing_images:
            file.write("\n## Images in labels but missing in images directory\n\n")
            for image_name in missing_images:
                file.write(f"- {image_name}\n")

        if extra_images:
            file.write("\n## Images in images directory but missing in labels\n\n")
            for image_name in extra_images:
                file.write(f"- {image_name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and normalize OCR labels for meter readings."
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels CSV file.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Path to directory with images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labels_normalized.csv"),
        help="Path to save normalized labels.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("labels_validation_report.md"),
        help="Path to save validation report.",
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=None,
        help="Optional number of digits after dot. Example: --decimals 2",
    )

    parser.add_argument(
        "--allow-extra-images",
        action="store_true",
        help="Do not fail if images directory contains images missing from labels.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_and_normalize_labels(
        labels_path=args.labels,
        images_dir=args.images_dir,
        output_path=args.output,
        report_path=args.report,
        decimals=args.decimals,
        fail_on_extra_images=not args.allow_extra_images,
    )


if __name__ == "__main__":
    main()

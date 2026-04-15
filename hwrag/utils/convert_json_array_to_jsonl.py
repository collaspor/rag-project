import argparse
import json
from pathlib import Path


def convert_json_array_to_jsonl(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input file must be a JSON array: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(data):
            if item is None:
                continue
            if not isinstance(item, dict):
                raise ValueError(
                    f"Item at index {index} is not a JSON object: {type(item).__name__}"
                )

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            valid_count += 1

    return valid_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSON array file into JSONL format."
    )
    parser.add_argument("input", help="Path to the source JSON array file.")
    parser.add_argument("output", help="Path to the target JSONL file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    count = convert_json_array_to_jsonl(input_path, output_path)
    print(f"Converted {count} records to JSONL: {output_path}")


if __name__ == "__main__":
    main()

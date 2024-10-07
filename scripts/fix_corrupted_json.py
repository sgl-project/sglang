import json
import re
import sys


def clean_json_file(input_file, output_file):
    try:
        # Open the input file with 'replace' option for handling bad characters
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()

        # Replace bad characters (represented by '�' after decoding) with a space
        cleaned_data = data.replace("�", " ")

        # Remove control characters (e.g., ASCII control characters like \x00 to \x1F)
        # These can cause issues in JSON parsing.
        cleaned_data = re.sub(r"[\x00-\x1F]+", " ", cleaned_data)

        # Parse cleaned data as JSON
        json_data = json.loads(cleaned_data)

        # Write the cleaned JSON to a new output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"Cleaned JSON file has been saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    assert len(sys.argv) > 1, "please give the input file path"
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = output_file = sys.argv[1]

    clean_json_file(input_file, output_file)

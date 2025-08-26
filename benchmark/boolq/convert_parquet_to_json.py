import sys

import pyarrow.parquet as pq


def convert_parquet_to_json(input_file, output_file):
    # read parquet file
    table = pq.read_table(input_file)

    # turn parquet data to dataframe
    df = table.to_pandas()

    # turn dataframe to json form
    json_data = df.to_json(orient="records", lines=True)

    # write json to file
    with open(output_file, "w") as f:
        f.write(json_data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:python convert_parquet_to_json.py <input_file> <output_file>")

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_parquet_to_json(input_file, output_file)

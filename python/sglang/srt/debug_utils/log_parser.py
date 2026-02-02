_PATTERN_DECODE = (
    r"(\(\w+ pid=(?P<pid>\d+)(?:,\s*ip=(?P<ip>[\d\.]+))?\))?\s*"
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:\s+DP(?P<dp_rank>\d+))?"
    r"(?:\s+TP(?P<tp_rank>\d+))?"
    r"(?:\s+EP(?P<ep_rank>\d+))?"
    r"(?:\s+PP(?P<pp_rank>\d+))?"
    r"\]\s+"
    r"Decode batch( \[\d+\])?,\s+"
    r"#running-req:\s*(?P<num_running_req>\d+),\s+"
    r"#token:\s*(?P<num_token>\d+),\s+"
    r"token usage:\s*(?P<token_usage>[0-9.]+),\s+"
    r".*?"
    r"gen throughput \(token/s\):\s*(?P<gen_throughput>[0-9.]+),\s+"
    r"#queue-req:\s*(?P<queue_req>\d+),"
)


def parse(lines):
    import polars as pl

    df = pl.DataFrame(dict(line=lines.splitlines()))
    df = df.with_columns(info=pl.col("line").str.extract_groups(_PATTERN_DECODE))
    df = df.unnest("info")
    df = df.filter(pl.col("gen_throughput").is_not_null())

    df = df.with_columns(
        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        *[
            pl.col(col).cast(dtype)
            for col, dtype in [
                ("pid", pl.Int64),
                ("dp_rank", pl.Int64),
                ("tp_rank", pl.Int64),
                ("ep_rank", pl.Int64),
                ("pp_rank", pl.Int64),
                ("num_running_req", pl.Int64),
                ("num_token", pl.Int64),
                ("token_usage", pl.Float64),
                ("gen_throughput", pl.Float64),
                ("queue_req", pl.Int64),
            ]
            if col in df.columns
        ],
    )
    return df

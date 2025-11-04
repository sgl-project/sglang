_PATTERN_DECODE = (
    r"(\(SGLangEngine pid=(?P<pid>\d+)(?:,\s*ip=(?P<ip>[\d\.]+))?\))?\s+"
    r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:\s+DP(?P<dp>\d+))?"
    r"(?:\s+TP(?P<tp>\d+))?"
    r"(?:\s+PP(?P<pp>\d+))?"
    r"\]\s+"
    r"Decode batch( \[\d+\])?,\s+"
    r"#running-req:\s*(?P<running_req>\d+),\s+"
    r"#token:\s*(?P<token>\d+),\s+"
    r"token usage:\s*(?P<token_usage>[0-9.]+),\s+"
    r".*?"
    r"gen throughput \(token/s\):\s*(?P<throughput>[0-9.]+),\s+"
    r"#queue-req:\s*(?P<queue>\d+),"
)


def parse(lines):
    import polars as pl

    df = pl.DataFrame(dict(line=lines.splitlines()))
    df = df.with_columns(info=pl.col('line').str.extract_groups(_PATTERN_DECODE))
    df = df.unnest('info')
    df = df.filter(pl.col('throughput').is_not_null())

    df = df.with_columns(
        pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        *[
            pl.col(c).cast(pl.Int64)
            for c in [
                "pid",
                "dp",
                "tp",
                "pp",
                "running_req",
                "token",
                "token_usage",
                "throughput",
                "queue",
            ]
            if c in df.columns
        ],
    )
    return df

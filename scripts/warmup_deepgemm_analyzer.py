import polars as pl
from sglang.srt import warmup_deepgemm


def run():
    df = pl.DataFrame(warmup_deepgemm.read_output())
    print(df)


if __name__ == '__main__':
    run()

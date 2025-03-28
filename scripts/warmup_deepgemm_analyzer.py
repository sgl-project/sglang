import polars as pl
from sglang.srt import warmup_deepgemm

df_raw = pl.DataFrame(warmup_deepgemm.read_output())
print(df_raw)

df = df_raw.group_by('k', 'n').agg(pl.col('m').unique().sort())
with pl.Config(fmt_str_lengths=1000, fmt_table_cell_list_len=1000, tbl_cols=-1, tbl_rows=-1):
    print(df)

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def plot_simulation_output_df_step(df_step: pl.DataFrame):
    df_step = df_step.to_pandas()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df_step['step'], y=df_step['utilization_rate'], name='Utilization Rate'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df_step['step'], y=df_step['num_tokens_of_batch'], name='Num Tokens of Batch'),
        secondary_y=True,
    )
    fig.show()

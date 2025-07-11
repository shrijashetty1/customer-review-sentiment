import plotly.graph_objects as go
import pandas as pd

# Display week by week evolition of main scores
def weekEvolution(reviews, label_mapping):
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
    limit_date = reviews['date'].max()
    reviews['week'] = reviews['date'] - pd.to_timedelta(reviews['date'].dt.weekday, unit='d')
    reviews['week'] = reviews['week'].dt.strftime('%Y-%m-%d')

    last_weeks = reviews[reviews['date'] >= limit_date - pd.DateOffset(weeks=4)]
    weekly_avg_scores = last_weeks.groupby('week')[list(label_mapping.keys())].mean()

    fig_line = go.Figure()
    color_line = ['#32CD32', 'rgba(31, 119, 180, 0.8)', 'rgba(107, 174, 214, 0.8)', 'rgba(158, 202, 225, 0.8)'] 

    for i, column in enumerate(weekly_avg_scores.columns):
        label = label_mapping[column]
        fig_line.add_trace(
            go.Scatter(
                x=weekly_avg_scores.index,
                y=weekly_avg_scores[column],
                mode='lines+markers',
                name=label,
                line=dict(color=color_line[i]),
                text=[f"{label}: {val:.2f}" for val in weekly_avg_scores[column]],
                hoverinfo="text"
            )
        )

    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False, title_text='Average Score')
    fig_line.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        height=250, width=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.65,
            xanchor="center",
            x=0.5
        )
    )

    return fig_line
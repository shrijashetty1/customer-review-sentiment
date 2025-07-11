import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Plot a basic resume of the KPIs
def plotAverageScoresAndReviews(reviews, resumme_raw, app=False):
    # Calculate the average for each score
    average_food = reviews['food_score'].mean()
    average_service = reviews['service_score'].mean()
    average_atmosphere = reviews['atmosphere_score'].mean()
    average_reviews = (resumme_raw['stars'] * resumme_raw['reviews']).sum() / resumme_raw['reviews'].sum()

    # Create a figure with horizontal subplots
    fig = make_subplots(rows=1, cols=3, 
                        specs=[[{"type": "xy"}, {"type": "bar"}, {"type": "bar"}]], 
                        subplot_titles=("Average Score", "Number of Reviews", "Categories"))

    # First subplot: Display the average review as large text
    fig.add_trace(
        go.Scatter(x=[0], y=[0], text=[f"{average_reviews:.2f}"], mode="text", textfont=dict(size=120)),
        row=1, col=1
    )

    # Remove grid, lines, and ticks from the first subplot
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    # Second subplot: Bar plot for reviews
    fig.add_trace(
        go.Bar(x=resumme_raw['reviews'], y=resumme_raw['stars'], marker=dict(color='lightskyblue'),
               text=resumme_raw['reviews'], textposition='auto', name="Reviews", orientation='h'),
        row=1, col=2
    )

    # Third subplot: Bar plot for categories (Food, Service, Atmosphere)
    fig.add_trace(
        go.Bar(x=[average_food, average_service, average_atmosphere], 
               y=['Food', 'Service', 'Atmosphere'], 
               marker=dict(color='lightgreen'), 
               text=[f"{average_food:.2f}", f"{average_service:.2f}", f"{average_atmosphere:.2f}"], 
               textposition='auto', 
               orientation='h', 
               name="Categories"),
        row=1, col=3
    )

    # Update layout for the whole figure
    fig.update_layout(height=500, width=1200, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)

    # Show the figure
    if app:
        return fig
    else:
        fig.show()

# Plot a basic views of the KPIs
def plotScoreTrends(reviews, app = False):
    # Convert date column to datetime format and create additional time columns
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
    reviews['month'] = reviews['date'].dt.to_period('M')
    reviews['year'] = reviews['date'].dt.year
    reviews['week'] = reviews['date'] - pd.to_timedelta(reviews['date'].dt.weekday, unit='d')
    reviews['week'] = reviews['week'].dt.strftime('%Y-%m-%d')

    # Filter data for the last periods (months, years, weeks)
    limit_date = reviews['date'].max()
    last_months = reviews[reviews['date'] >= limit_date - pd.DateOffset(months=12)]
    last_years = reviews[reviews['date'] >= limit_date - pd.DateOffset(years=8)]
    last_weeks = reviews[reviews['date'] >= limit_date - pd.DateOffset(weeks=5)]

    # Update the axis labels for each score to be more readable
    label_mapping = {
        'rating_score': 'Rating',
        'food_score': 'Food',
        'service_score': 'Service',
        'atmosphere_score': 'Atmosphere'
    }

    # Compute averages for the required periods
    monthly_avg_scores = last_months.groupby('month')[['rating_score', 'food_score', 'service_score', 'atmosphere_score']].mean()
    yearly_avg_scores = last_years.groupby('year')[['rating_score']].mean()
    weekly_avg_scores = last_weeks.groupby('week')[['rating_score', 'food_score', 'service_score', 'atmosphere_score']].mean()

    # Create a figure with subplots using the Z-layout
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"colspan": 2}, None],
                               [{}, {}]],  # 1 large plot on the first row, 2 smaller plots on the second
                        subplot_titles=("Monthly Score Trends (Last 12 Months)", 
                                        "Annual Rating Score Trends (Last 6 Years)", 
                                        "Weekly Score Trends (Last 4 Weeks)"))

    # Add monthly score trends to the first row
    colors = ['#1f77b4', '#aec7e8', '#aec7e8', '#aec7e8']
    for i, column in enumerate(monthly_avg_scores.columns):
        label = label_mapping[column]
        fig.add_trace(
            go.Scatter(x=monthly_avg_scores.index.astype(str), y=monthly_avg_scores[column],
                       mode='lines+markers', name=label, 
                       text=[f"{label} - {val:.2f}" for val in monthly_avg_scores[column]], 
                       hoverinfo="text", line=dict(color=colors[i])),
            row=1, col=1)

    # Add yearly score trends to the second row (left)
    fig.add_trace(
        go.Scatter(x=yearly_avg_scores.index.astype(str), y=yearly_avg_scores['rating_score'],
                   mode='lines+markers', name="Rating", line=dict(color='#1f77b4', width=4),
                   text=[f"Rating - {val:.2f}" for val in yearly_avg_scores['rating_score']], 
                   hoverinfo="text"),
        row=2, col=1)

    # Add weekly score trends to the second row (right)
    for i, column in enumerate(weekly_avg_scores.columns):
        label = label_mapping[column]
        fig.add_trace(
            go.Scatter(x=weekly_avg_scores.index.astype(str), y=weekly_avg_scores[column],
                       mode='lines+markers', name=label, 
                       text=[f"{label} - {val:.2f}" for val in weekly_avg_scores[column]], 
                       hoverinfo="text", line=dict(color=colors[i])),
            row=2, col=2)

    # Customize layout
    fig.update_layout(showlegend=False, 
                      title="Score Trends Analysis",
                      title_font=dict(size=28),
                      margin=dict(l=50, r=50, t=100, b=50),
                      paper_bgcolor="white",
                      height=800, width=1200)
    fig.update_xaxes(showline=False, showgrid=False)
    fig.update_yaxes(showline=False, showgrid=True)

    # Customize x-axes formatting
    fig.update_xaxes(tickformat="%Y", row=2, col=1)  # Yearly format
    fig.update_xaxes(tickformat="%d-%b", row=2, col=2)  # Weekly format

    # Add annotations
    fig.add_annotation(x='2024-06', y=4.8, text="Highest Score", showarrow=True, arrowhead=2, ax=0, ay=80, row=1, col=1)
    fig.add_annotation(x='2024-03', y=4.5, text="Drop in March", showarrow=True, arrowhead=2, ax=0, ay=-40, row=1, col=1)
    fig.add_annotation(x='2024-08', y=4.5, text="Drop in August", showarrow=True, arrowhead=2, ax=0, ay=-40, row=1, col=1)

    # Update marker sizes
    fig.update_traces(marker=dict(size=8), selector=dict(name="Rating"))
    
    # Show the figure
    if app:
        return fig
    else:
        fig.show()

# Plot the evolution of distribution of reviews on time based on sentiments
def plotSentimentTrend(df, years_limit = 2, app = False):
    # Convert date to datetime format and handle missing values
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Filter only the last 6 years
    last_six_years = df['date'].max() - pd.DateOffset(years=years_limit)
    df = df[df['date'] >= last_six_years]

    # Set date as index for resampling
    df.set_index('date', inplace=True)
    
    # Resample to monthly and count sentiments
    sentiment_counts = df.resample('M')['sentiment_label'].value_counts().unstack().fillna(0)

    # Calculate the percentage for each sentiment type
    sentiment_percentage = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    sentiment_percentage = sentiment_percentage.round(2)
    sentiment_percentage = sentiment_percentage.reset_index().melt(id_vars=['date'], value_name='percentage', var_name='sentiment_label')
    
    # Plot sentiment percentage evolution
    fig = px.area(
        sentiment_percentage,
        x='date',
        y='percentage',
        color='sentiment_label',
        #title='Sentiment Percentage Over the Last ' + str(years_limit) + ' Years',
        labels={'date': '', 'percentage': 'Percentage of Reviews (%)', 'sentiment_label': 'Sentiment'},
        template='plotly_white',
    )

    # Customize layout
    fig.update_layout(
        #title=dict(x=0.5, xanchor='center', font=dict(size=18, color='black')),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, title='Percentage of Reviews', ticksuffix='%'),
        legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        width=1200,
        height=400
    )

    # Customize color for sentiment categories
    color_map = {
        'positive': 'rgba(102, 194, 165, 0.7)', 
        'neutral': 'rgba(141, 160, 203, 0.7)', 
        'negative': 'rgba(252, 141, 98, 0.7)'
    }
    fig.for_each_trace(lambda trace: trace.update(line=dict(width=0, shape='spline'), fill='tonexty', fillcolor=color_map.get(trace.name, 'rgba(150, 150, 150, 0.5)')))

    # Remove the plot frame and keep the visualization as clean as possible
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showline=False, range=[0, 100])  # Percentage scale from 0 to 100

    if app:
        return fig
    else:
        fig.show()

# Compute k-nearest neighbors
def plotKdistance(reduced_embeddings, k=5, method='PCA', app = False):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(reduced_embeddings)
    distances, _ = neighbors_fit.kneighbors(reduced_embeddings)
    
    # Sort distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])
    
    # Create interactive line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(1, len(k_distances) + 1),
        y=k_distances,
        mode='lines',
        line=dict(color='blue'),
        name='k-distance'
    ))
    
    # Update layout for clarity
    fig.update_layout(
        title=f'k-Distance Graph for {method}',
        xaxis_title='Points sorted by distance',
        yaxis_title=f'Distance to {k}th Nearest Neighbor',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Add light grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    if app:
        return fig
    else:
        fig.show()

# Plot reviews by communities, using embeddingsm cosine_similarity and Girvan-Newman algorithm
def plotCommunities(reviews, app = False):
    # Load embeddings from reviews
    ebm_reviews = np.array(reviews['embedding'].tolist())

    # Calculate cosine similarity matrix between all pairs of embeddings
    similarity_matrix = cosine_similarity(ebm_reviews)
    similarity_threshold = 0.75

    G_sparser = nx.Graph()

    # Add nodes representing each review
    for i in range(len(reviews)):
        G_sparser.add_node(i, sentiment_label=reviews['sentiment_label'].iloc[i])

    # Add edges based on the similarity matrix and new threshold
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):  # Only consider upper triangle to avoid redundancy
            if similarity_matrix[i][j] >= similarity_threshold:
                G_sparser.add_edge(i, j, weight=similarity_matrix[i][j])

    # Use Girvan-Newman algorithm to detect communities
    comp = nx.algorithms.community.girvan_newman(G_sparser)
    communities_sparser = tuple(sorted(c) for c in next(comp))

    # Extract key terms from each community using TF-IDF
    vectorizer = TfidfVectorizer(max_features=3, stop_words='english')
    community_keywords = []

    for community in communities_sparser:
        reviews_text = reviews.iloc[list(community)]['cleaned_review'].astype(str).tolist()
        # Ensure there are non-stopword terms to avoid empty vocabulary error
        filtered_reviews_text = [text for text in reviews_text if len(vectorizer.build_tokenizer()(text)) > 0]
        if len(filtered_reviews_text) > 1:
            tfidf_matrix = vectorizer.fit_transform(filtered_reviews_text)
            keywords = vectorizer.get_feature_names_out()
            community_keywords.append(", ".join(keywords))
        else:
            community_keywords.append(reviews.iloc[list(community)[0]]['cleaned_review'])

    # Prepare data for Plotly interactive visualization
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    pos = nx.spring_layout(G_sparser, seed=42)
    colors = px.colors.qualitative.Set1  # A set of distinct colors for different communities

    # Extract node positions, colors, and labels for Plotly
    for i, community in enumerate(communities_sparser):
        for node in community:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(colors[i % len(colors)])
            node_text.append(f"{community_keywords[i]}")

    # Create edge traces
    edge_x = []
    edge_y = []

    for edge in G_sparser.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create the Plotly figure
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=10,
            line_width=2,
            color=node_color
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        #title='Reviews by Communities',
                        #titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    if app:
        return fig
    else:
        fig.show()
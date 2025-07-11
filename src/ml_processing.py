import pandas as pd
import numpy as np
import re
import ast
import json

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

from transformers import pipeline
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from gensim import corpora
from gensim.models import LdaModel
from sklearn.decomposition import PCA
import umap.umap_ as umap

import plotly.express as px

# Download NLTK stopwords and lexicon
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Clean text, stopworks and tokenize words
def clean_text(text):
    # Load spaCy Spanish model
    nlp = spacy.load('es_core_news_sm')

    text = text.lower()
    text = re.sub(r'[^a-záéíóúñü0-9\s]', '', text)
    
    doc = nlp(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [token.lemma_ for token in doc 
              if token.text not in stop_words and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

# Extract sentiment for each review using 
def analyzeSentiment(df, score_colum = 'rating_score',):
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to each review using VADER
    df['vader_sentiment'] = df['cleaned_review'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Classify sentiment into positive, neutral, negative using rating_score and vader_sentiment
    def classify_sentiment(row, score_colum = score_colum):
        if row[score_colum] >= 4:
            return 'positive'
        elif row[score_colum] <= 2:
            return 'negative'
        elif row['vader_sentiment'] > 0.05:
            return 'positive'
        elif row['vader_sentiment'] < -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    df['sentiment_label'] = df.apply(classify_sentiment, axis=1)
    
    return df

# Extract most common words for a selected sentiment
def extractCommonWords(df, sentiment_label='positive', n=10):
    # Filter reviews by sentiment label
    filtered_reviews = df[df['sentiment_label'] == sentiment_label]['cleaned_review'].fillna('').tolist()
    
    # Tokenize and count words for the given sentiment label
    vectorizer = CountVectorizer().fit(filtered_reviews)
    word_counts = vectorizer.transform(filtered_reviews).sum(axis=0)
    
    # Create a dictionary of word frequencies
    word_freq = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)[:n]
    
    return sorted_word_freq

# Extract most common n-grams for a selected sentiment
def extractCommonNgrams(df, sentiment_label='positive', n=2, top_n=10):
    # Filter reviews by sentiment label
    filtered_reviews = df[df['sentiment_label'] == sentiment_label]['cleaned_review'].fillna('').tolist()
    
    # Create n-grams for the given sentiment label
    vectorizer = CountVectorizer(ngram_range=(n, n)).fit(filtered_reviews)
    ngram_counts = vectorizer.transform(filtered_reviews).sum(axis=0)
    
    # Create a list of n-grams with their counts
    ngram_freq = [(word, ngram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_ngrams = sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:top_n]
    
    return sorted_ngrams

# Extract most and least recommendations mentioned
def analyzeRecommendations(df):
    all_dishes = []

    # Convert string representation of lists to actual lists and extend all_dishes
    for item in df['recommendations_list'].dropna():
        try:
            dishes = ast.literal_eval(item)
            if isinstance(dishes, list):
                all_dishes.extend(dishes)
        except:
            continue

    # Filter out empty values
    all_dishes = [dish for dish in all_dishes if dish.strip() != '']

    # Count the frequency of each dish
    dish_counts = Counter(all_dishes)
    if not dish_counts:
        return [], []
    
    # Most and least recommended dishes
    most_common_dishes = dish_counts.most_common(3)
    min_count = min(dish_counts.values())
    worst_dishes = [dish for dish, count in dish_counts.items() if count == min_count]

    return most_common_dishes, worst_dishes

# Extract the embeddings for each cleaned review
def get_embedding(text):
    # Import Bert model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to apply DBSCAN
def apply_dbscan(reduced_embeddings, eps=0.6, min_samples=5):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(reduced_embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled)
    return labels

# Extract topics using LDA model
def analyzeTopicsLDA(df, number_of_topics = 5):
   # Prepare corpus for LDA
    cleaned_reviews = df['cleaned_review'].dropna().tolist()
    tokenized_reviews = [review.split() for review in cleaned_reviews if isinstance(review, str) and review.strip() != '']
    
    if not tokenized_reviews:
        print("No valid reviews to process.")
        return None, []
    
    dictionary = corpora.Dictionary(tokenized_reviews)
    if len(dictionary) == 0:
        print("Dictionary is empty after tokenization.")
        return None, []
    
    corpus = [dictionary.doc2bow(review) for review in tokenized_reviews]
    if not any(corpus):
        print("Corpus is empty. No terms found in any document.")
        return None, []
    
    # Train LDA model
    try:
        lda_model = LdaModel(
            corpus,
            num_topics=number_of_topics,
            id2word=dictionary,
            passes=10,
            random_state=42
        )
    except ValueError as e:
        print(f"LDA Model training failed: {e}")
        return None, []
    
    # Extract topics
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(f"Topic {topic[0]}: {topic[1]}")
    return lda_model, topics

# Generate topics for all selected columns in group columns
def generateTopicsbyColumn(reviews, group_columns):
    # Initialize dictionary to store topics
    topics_dict = {group_col: {} for group_col in group_columns}

    # Iterate over each grouping column and generate topics
    for group_col in group_columns:
        print(f"\n=== Topics by {group_col} ===")
        unique_groups = reviews[group_col].dropna().unique()
        
        for group_val in unique_groups:
            subset = reviews[reviews[group_col] == group_val]
            
            # Check if there are enough reviews to train LDA
            if len(subset) < 5:
                print(f"\n--- {group_col} = {group_val} ---")
                print("Not enough data to train LDA.")
                continue
            
            print(f"\n--- {group_col} = {group_val} ---")
            
            # Generate topics for the current subset
            lda_model, topics = analyzeTopicsLDA(subset)
            
            if lda_model is not None and topics:
                # Store topics as strings in the dictionary
                topics_strings = [topic[1] for topic in topics]
                topics_dict[group_col][group_val] = topics_strings
            else:
                print("No topics generated for this group.\n")
    return topics_dict

# Extract the periods with less score and the reviews of each period
def analyzeLowScores(df, score_column = 'rating_score', time_period='month', num_periods=1, last_periods = 12):
    # Generate extra granularity
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M')
    df['year'] = df['date'].dt.year

    # Calculate the mean and standard deviation of the scores
    last_periods = df[df['date'] >= df['date'].max() - pd.DateOffset(months=last_periods)]

    # Compute averages for the required periods
    last_periods_avg_scores = last_periods.groupby(time_period)[score_column].mean().reset_index()
    last_periods_avg_scores.set_index(time_period, inplace=True)
    
    mean_score = last_periods_avg_scores[score_column].mean()
    std_dev_score = last_periods_avg_scores[score_column].std()
    
    # Define a threshold for low scores
    threshold = mean_score - std_dev_score
    low_scores = last_periods_avg_scores[last_periods_avg_scores[score_column] < threshold]
    # Select the specified number of periods with the lowest average score
    low_score_periods = low_scores.index[:num_periods]
    
    # Filter negative reviews for the selected periods with the lowest score
    period_reviews = df[(df[time_period].isin(low_score_periods)) & 
                        (df[score_column] <= 3)]
    
    # Drop the 'embedding' column if it exists to avoid issues with non-hashable types
    if 'embedding' in period_reviews.columns:
        period_reviews = period_reviews.drop(columns=['embedding'])
    
    # Add a column indicating the period with the lowest score for easier filtering
    period_reviews['low_score_period'] = period_reviews[time_period]
    period_reviews = period_reviews.sort_values('low_score_period')

    return period_reviews, low_score_periods

# Calculate topics for each low_score_period and concatenate results
def generateTopicsPerPeriod(df, score_column = 'rating_score', number_of_topics=1):
    valid_reviews = df[df['review'].notna()]
    topics_dict = {score_column: {}}
    for period in valid_reviews['low_score_period'].unique():
        period_reviews = valid_reviews[valid_reviews['low_score_period'] == period]
        # Assuming analyzeTopicsLDA function returns topics as the second output
        _, topics = analyzeTopicsLDA(period_reviews, number_of_topics=number_of_topics)
        topics_dict[score_column][period] = topics
    return topics_dict

# Format arrays of words in json format
def format_words(words_list):
    return {str(word): int(weight) if isinstance(weight, (int, np.integer)) else weight for word, weight in words_list}

# UMAP Embeddings Visualization
def calculateAndVisualizeEmbeddingsUMAP(df, plot = True, app = False):
    embeddings = np.array(df['embedding'].tolist())
    sentiment_labels = df['sentiment_label']

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create DataFrame for visualization
    viz_df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    viz_df['sentiment_label'] = sentiment_labels

    # Scatter plot with Plotly for interactive visualization
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='sentiment_label',
        title='Embedding Visualization with UMAP',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'},
        opacity=0.7
    )
    fig.update_layout(
        showlegend=True, legend=dict(title='Sentiment'),
        margin=dict(l=0, r=10, t=30, b=10),
        width=700, 
        height=500
    )
    
    if plot:    
        fig.show()

    if app:
        return reduced_embeddings, fig
    else:
        return reduced_embeddings

# PCA Embeddings Visualization
def calculateAndVisualizeEmbeddingsPCA(df, score_column = 'rating_score', plot = True, app = False):
    # Convert embeddings to a NumPy array
    embeddings = np.array(df['embedding'].tolist())
    ratings = df[score_column]
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Calculate variance explained by each component
    var_explained = pca.explained_variance_ratio_ * 100
    var1, var2 = var_explained
    
    # Prepare DataFrame for Plotly
    plot_df = pd.DataFrame({
        'PCA Component 1': reduced_embeddings[:, 0],
        'PCA Component 2': reduced_embeddings[:, 1],
        'Rating Score': ratings,
        'Review ID': df.get('review_id', range(len(df)))  # Optional identifier
    })
    # Create interactive scatter plot
    fig = px.scatter(
        plot_df,
        x='PCA Component 1',
        y='PCA Component 2',
        color='Rating Score',
        color_continuous_scale='Viridis',
        hover_data=['Review ID', 'Rating Score'],
        title=f'Embeddings by Rating Score (PCA 1: {var1:.1f}%, PCA 2: {var2:.1f}%)',
        labels={
            'PCA Component 1': f'PCA 1 ({var1:.1f}% variance)',
            'PCA Component 2': f'PCA 2 ({var2:.1f}% variance)',
            'Rating Score': 'Rating Score'
        }
    )
    # Enhance layout for clarity
    fig.update_layout(
        template='plotly_white',
        coloraxis_colorbar=dict(
            title='Rating Score',
            tickmode='linear'
        ),
        hovermode='closest',
        margin=dict(l=0, r=10, t=30, b=10),
        width=700, 
        height=500
    )
        
    if plot:    
        fig.show()

    if app:
        return reduced_embeddings, fig
    else:
        return reduced_embeddings

# PCA Visualization with DBSCAN
def calculateAndVisualizeEmbeddingsPCA_with_DBSCAN(df, score_column = 'rating_score', eps=0.55, min_samples=10, plot = True, app = False):
    embeddings = np.array(df['embedding'].tolist())
    ratings = df[score_column]
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    var1, var2 = pca.explained_variance_ratio_ * 100
    
    clusters = apply_dbscan(reduced, eps, min_samples)
    
    plot_df = pd.DataFrame({
        'pca_component_1': reduced[:, 0],
        'pca_component_2': reduced[:, 1],
        score_column: ratings,
        'pca_cluster': clusters,
        'review_id': df.get('review_id', range(len(df)))
    })

    fig = px.scatter(
        plot_df,
        x='pca_component_1',
        y='pca_component_2',
        color='pca_cluster',
        color_continuous_scale='Viridis',
        hover_data=['review_id', score_column],
        title=f'PCA with DBSCAN (PCA1: {var1:.1f}%, PCA2: {var2:.1f}%)',
        labels={
            'PCA 1': f'pca_component_1 ({var1:.1f}% variance)',
            'PCA 2': f'pca_component_2 ({var2:.1f}% variance)',
            'Cluster': 'pca_cluster'
        }
    )
    
    fig.update_layout(
        template='plotly_white',
        coloraxis_colorbar=dict(title='pca_cluster'),
        hovermode='closest',
        margin=dict(l=0, r=10, t=30, b=10),
        width=700, 
        height=500
    )

    if plot:    
        fig.show()

    if app:
        return plot_df, fig
    else:
        return plot_df

# UMAP Visualization with DBSCAN
def calculateAndVisualizeEmbeddingsUMAP_with_DBSCAN(df, eps=0.7, min_samples=10, plot = True, app = False):
    embeddings = np.array(df['embedding'].tolist())
    sentiment = df['sentiment_label']
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    
    clusters = apply_dbscan(reduced, eps, min_samples)
    
    plot_df = pd.DataFrame({
        'umap_component_1': reduced[:, 0],
        'umap_component_2': reduced[:, 1],
        'sentiment': sentiment,
        'umap_cluster': clusters,
        'review_id': df.get('review_id', range(len(df)))
    })

    fig = px.scatter(
        plot_df,
        x='umap_component_1',
        y='umap_component_2',
        color='umap_cluster',
        color_continuous_scale='Viridis',
        hover_data=['sentiment', 'umap_cluster'],
        title='UMAP with DBSCAN',
        labels={
            'UMAP 1': 'umap_component_1',
            'UMAP 2': 'umap_component_2',
            'Cluster': 'umap_cluster'
        },
        opacity=0.7
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(title='umap_cluster'),
        hovermode='closest',
        margin=dict(l=0, r=10, t=30, b=10),
        width=700, 
        height=500
    )

    if plot:    
        fig.show()
        
    if app:
        return plot_df, fig
    else:
        return plot_df
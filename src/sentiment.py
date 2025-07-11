import pandas as pd
import numpy as np
import argparse
import sys
import os
import json

from tqdm import tqdm

import ml_processing
import plots
import llm_insights

## Load the processed and cleaned data
processed_data_path = '../data/processed/'
raw_data_path = '../data/raw/'
sys.path.append(os.path.abspath(os.path.join('..')))

# Label mapping for interest columns and label name
label_mapping = {
    'rating_score': 'Rating',
    'food_score': 'Food',
    'service_score': 'Service',
    'atmosphere_score': 'Ambient'
}

# Parameters
number_of_words = 10
n_grams = 2
eps = 0.5
min_samples = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sentiment analysis.')
    parser.add_argument('--name', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('--plot', type=bool, default=True, help='Whether to generate plots or not')

    args = parser.parse_args()

    name = args.name
    plot = args.plot

    reviews_pro = pd.read_csv(processed_data_path + name + '_reviews.csv')
    resumme_raw = pd.read_csv(raw_data_path + 'resumme_' + name + '.csv')

    print(resumme_raw)
    print(reviews_pro.sample(5))

    reviews = reviews_pro.copy()
    reviews.reset_index(drop=True, inplace=True)
    resumme = resumme_raw.copy()

    ## Cleaning and preprocessing
    tqdm.pandas(desc="Cleaning Reviews")
    reviews['cleaned_review'] = reviews['review'].fillna('').progress_apply(ml_processing.clean_text)

    print(reviews[['review', 'cleaned_review']].sample(5))


    label_keys = list(label_mapping.keys())

    ## Analyze sentiment
    # Analyze sentiment with VADER
    reviews = ml_processing.analyzeSentiment(reviews)

    # Extract common positive and negative phrases
    common_positive_words = ml_processing.extractCommonWords(reviews, sentiment_label = 'positive', n = number_of_words)
    common_negative_words = ml_processing.extractCommonWords(reviews, sentiment_label = 'negative', n = number_of_words)

    print("Top Positive Words:", common_positive_words)
    print("Top Negative Words:", common_negative_words)

    # Extract common positive and negative bigrams
    common_positive_bigrams = ml_processing.extractCommonNgrams(reviews, sentiment_label='positive', n = n_grams, top_n=number_of_words)
    common_negative_bigrams = ml_processing.extractCommonNgrams(reviews, sentiment_label='negative', n = n_grams, top_n=number_of_words)

    print("Top Positive Bigrams:", common_positive_bigrams)
    print("Top Negative Bigrams:", common_negative_bigrams)

    if plot:
        plots.plotSentimentTrend(reviews, years_limit=2)

    #most_recommended, less_recommended = ml_processing.analyzeRecommendations(reviews)
    #print("Top Most Recommended:", most_recommended)
    #print("Least Recommended :", less_recommended)

    ## Calculate embeddings
    tqdm.pandas(desc="Generating Embeddings")
    reviews['embedding'] = reviews['cleaned_review'].progress_apply(ml_processing.get_embedding)

    ## Analyze embeddings
    embeddings_pca = ml_processing.calculateAndVisualizeEmbeddingsPCA(reviews, score_column = label_keys[0], plot = plot)
    embeddings_umap = ml_processing.calculateAndVisualizeEmbeddingsUMAP(reviews, plot)

    # Visualize with DBSCAN clusters
    pca_clusters = ml_processing.calculateAndVisualizeEmbeddingsPCA_with_DBSCAN(reviews, score_column = label_keys[0], eps=eps, min_samples=min_samples, plot = plot)
    umap_clusters = ml_processing.calculateAndVisualizeEmbeddingsUMAP_with_DBSCAN(reviews, eps=eps, min_samples=min_samples, plot = plot)

    ## Join PCA and UMAP clusters info to reviews
    reviews = reviews.reset_index().rename(columns={'index':'review_id'})
    reviews = reviews.merge(pca_clusters[['review_id','pca_cluster']]).merge(umap_clusters[['review_id','umap_cluster']])

    ## Save processed reviews
    reviews.to_csv(processed_data_path + name + '_ml_processed_reviews.csv', index=False)
    print('OK! -> processed sample reviews saved at', processed_data_path + name + '_ml_processed_reviews.csv')

    ## Topics
    print('=== General topics ===')
    lda_model, topics = ml_processing.analyzeTopicsLDA(reviews)

    group_columns = ['pca_cluster', 'umap_cluster', 'sentiment_label']
    topics_dict = ml_processing.generateTopicsbyColumn(reviews, group_columns)

    # Usage
    time_period = 'month'  # Change to 'week', 'year', etc. to analyze different periods
    num_periods = 3  # Number of periods with the lowest average score to select

    # Analyze for each score type
    negative_periods_rating_reviews, low_score_periods = ml_processing.analyzeLowScores(reviews, label_keys[0], time_period, num_periods)
    negative_periods_food_reviews, _ = ml_processing.analyzeLowScores(reviews, label_keys[1], time_period, num_periods)
    negative_periods_service_reviews, _ = ml_processing.analyzeLowScores(reviews, label_keys[2], time_period, num_periods)
    negative_periods_atmosphere_reviews, _ = ml_processing.analyzeLowScores(reviews, label_keys[3], time_period, num_periods)

    negative_periods_rating_topics = ml_processing.generateTopicsPerPeriod(negative_periods_rating_reviews, label_keys[0])
    negative_periods_food_topics = ml_processing.generateTopicsPerPeriod(negative_periods_food_reviews, label_keys[1])
    negative_periods_service_topics = ml_processing.generateTopicsPerPeriod(negative_periods_service_reviews, label_keys[2])
    negative_periods_atmosphere_topics = ml_processing.generateTopicsPerPeriod(negative_periods_atmosphere_reviews, label_keys[3])

    negative_periods_topics = {**negative_periods_rating_topics, **negative_periods_food_topics, **negative_periods_service_topics, **negative_periods_atmosphere_topics}

    ## Extract outliers and painpoints
    # Join all the available information
    words_dict = {
        "common_positive_words": ml_processing.format_words(common_positive_words),
        "common_negative_words": ml_processing.format_words(common_negative_words),
        "common_positive_bigrams": ml_processing.format_words(common_positive_bigrams),
        "common_negative_bigrams": ml_processing.format_words(common_negative_bigrams)
    }
    print(words_dict)

    reviews_summary_dict = {**topics_dict, **words_dict}
    print(reviews_summary_dict)

    ## Extract reviews samples
    # Calculate total score using the three main scores
    reviews_score = reviews.copy()
    food_score_mean = np.round(reviews_score[label_keys[1]].mean(), 2) / 5
    service_score_mean = np.round(reviews_score[label_keys[2]].mean(), 2) / 5
    atmosphere_score_mean = np.round(reviews_score[label_keys[3]].mean(), 2) / 5

    reviews_score[label_keys[1]] = reviews_score[label_keys[1]].fillna(food_score_mean)
    reviews_score[label_keys[2]] = reviews_score[label_keys[2]].fillna(service_score_mean)
    reviews_score[label_keys[3]] = reviews_score[label_keys[3]].fillna(atmosphere_score_mean)

    reviews_score['total_score'] = np.round(
        reviews_score[label_keys[0]] +
        (reviews_score[label_keys[1]]/5 + reviews_score[label_keys[2]]/5 + reviews_score[label_keys[3]]/5) / 3, 2)

    # Filter not null reviews
    valid_reviews = reviews_score[reviews_score['review'].notna()]

    # Select the best and worst reviews in general
    best_reviews = valid_reviews[valid_reviews['total_score'] > 5]
    worst_reviews = valid_reviews[valid_reviews['total_score'] < 2.5]

    recent_best_reviews = best_reviews.sort_values(by='date', ascending=False)
    print('last_positive_reviews')
    print(recent_best_reviews.review)
    recent_worst_reviews = worst_reviews.sort_values(by='date', ascending=False)
    print('\nlast_negative_reviews')
    print(recent_worst_reviews.review)

    best_reviews_sample = best_reviews.sort_values(by='total_score', ascending=False)
    print('\nbest_reviews_sample')
    print(best_reviews_sample.review)
    worst_reviews_sample = worst_reviews.sort_values(by='total_score', ascending=True)
    print('\nworst_reviews_sample')
    print(worst_reviews_sample.review)

    low_score_reviews = negative_periods_rating_reviews[negative_periods_rating_reviews['review'].notna()][['month','review',label_keys[0]]]
    print('\nlow_score_reviews')
    print(low_score_reviews)
    print(low_score_periods)

    # Join all the samples
    recent_best_reviews['sample_type'] = 'recent_best_reviews'
    recent_worst_reviews['sample_type'] = 'recent_worst_reviews'
    best_reviews_sample['sample_type'] = 'best_reviews_sample'
    worst_reviews_sample['sample_type'] = 'worst_reviews_sample'
    low_score_reviews['sample_type'] = 'low_score_reviews'

    combined_reviews = pd.concat([
        recent_best_reviews,
        recent_worst_reviews,
        best_reviews_sample,
        worst_reviews_sample,
        low_score_reviews
    ])

    # Save samples
    combined_reviews.reset_index(drop=True, inplace=True)
    combined_reviews.to_csv(processed_data_path + name + '_sample_selected_reviews.csv', index=False)
    print('OK! -> processed sample reviews saved at', processed_data_path + name + '_sample_selected_reviews.csv')

    ## Extract Insights with LLM
    client = llm_insights.initChatGPTClient()
    # General insights
    general_insights_prompt = (
        "I have this information extracted from LDA topics using clustering and sentiment analysis, including positive and negative terms, in JSON format.\n"
        "I want you to extract:\n"
        "- 3 positive points\n"
        "- 3 negative points\n"
        "- 3 improvement suggestions based on the negative points\n"
        "\n"
        "Each point should be a logical, simple, and concise sentence that provides value. Do not name specific terms or topics, but focus on delivering direct value to business stakeholders without ambiguity. If you mention something that didn't go well, give examples based on the information.\n"
        "Return the result in English in JSON format, ensuring it is easy to read in a notebook and standardized as follows:\n"
        "\n"
        "{best:['','',''], worst:['','',''], improve:['','','']}\n"
        "\n"
        "Ensure there are no contradictions between positive, negative, and improvement points.\n"
        "The information:\n"
    )
    print(reviews_summary_dict)

    insigths_summary_dict = llm_insights.extractInsightsWithLLM(reviews_summary_dict, general_insights_prompt, client)
    print(insigths_summary_dict)

    ## Save insights
    json_file_path = processed_data_path + name + '_general_insights.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(insigths_summary_dict, json_file, indent=4)
    print('OK! -> general insights saved at', json_file_path)

    # Worst periods insights
    negative_periods_insights_prompt = (
        "I have this information extracted from LDA topics using clustering and sentiment analysis, including positive and negative terms at specific moments, in JSON format.\n"
        "\n"
        "I want you to extract:\n"
        "- For each date:\n"
        "- N negative points\n"
        "- N improvement suggestions based on the negative points\n"
        "\n"
        "Each point should be a logical, simple, and concise sentence that provides value. Do not mention specific terms or topics, but focus on delivering direct value to business stakeholders without ambiguity. If you mention something that didn't go well, provide examples based on the information.\n"
        "Return the result in English in JSON format, ensuring it is easy to read in a notebook and standardized as follows:\n"
        "\n"
        "{date: {problems:[problem, problem...], improve:[improve,improve...]}, date:{problems:[problem, problem...], improve:[improve,improve...]}, ...}\n"
        "\n"
        "Make sure there are no contradictions between the points.\n"
        "\n"
        "The information:\n"
    )
    print(negative_periods_topics)

    insigths_summary_dict = llm_insights.extractInsightsWithLLM(negative_periods_topics, negative_periods_insights_prompt, client)
    print(insigths_summary_dict)

    ## Save insights
    json_file_path = processed_data_path + name + '_worst_periods_insights.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(insigths_summary_dict, json_file, indent=4)
    print('OK! -> worst periods insights saved at', json_file_path)
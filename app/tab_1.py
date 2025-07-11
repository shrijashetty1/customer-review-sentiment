import pandas as pd
import numpy as np
import re
import json
import os
import sys
import importlib

import plotly.graph_objects as go
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..')))
from src import plots
from src import ml_processing
from src import llm_insights

# Update necesary topics for insights extraction
def updateTopicsDict(reviews):
    # Extract common positive and negative phrases
    common_positive_words = ml_processing.extractCommonWords(reviews, sentiment_label = 'positive', n = 10)
    common_negative_words = ml_processing.extractCommonWords(reviews, sentiment_label = 'negative', n = 10)

    print("Top Positive Words:", common_positive_words)
    print("Top Negative Words:", common_negative_words)

    # Extract common positive and negative bigrams
    common_positive_bigrams = ml_processing.extractCommonNgrams(reviews, sentiment_label='positive', n = 2, top_n=10)
    common_negative_bigrams = ml_processing.extractCommonNgrams(reviews, sentiment_label='negative', n = 2, top_n=10)

    group_columns = ['pca_cluster', 'umap_cluster', 'sentiment_label']
    topics_dict = ml_processing.generateTopicsbyColumn(reviews, group_columns)

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

    return reviews_summary_dict
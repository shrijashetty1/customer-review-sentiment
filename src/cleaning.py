import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def extractRestaurantDetailsFromReview(sample, search_words=None, verbose=False):
    # Takes a review text and applies regex to extract specific details 
    # (like service, price range, food score) based on the provided search patterns.
    
    clean_text = re.sub(r'\\ue[0-9a-f]{3}', '', sample)
    clean_text = re.sub(r'\n+', '\n', clean_text)
    clean_text = clean_text.strip()

    # Store extracted values
    extracted_values = []

    # Loop through search words to extract values dynamically
    for key, regex in search_words.items():
        match = re.search(regex, clean_text)
        value = match.group(1) if match else ''
        extracted_values.append(value)

    return extracted_values

def applyExtractDetails(df, search_words=None):
    # Applies the extraction function to the entire DataFrame, creating new columns 
    # for the extracted details based on the regex patterns provided.

    column_names = list(search_words.keys())
    df[column_names] = df['text_backup'].apply(lambda x: pd.Series(extractRestaurantDetailsFromReview(x, search_words=search_words)))
    return df

def extractReviewCount(text):
    # Extracts the number of reviews from a string (if present) and returns it as an integer.

    if isinstance(text, str):  # Verify if its a string
        match = re.search(r'(\d+)\s+reseñas', text)
        if match:
            return int(match.group(1))
    return None

def extractStarRating(text):
    # Extracts the star rating (out of 5) from a review string and returns it as an integer.

    match = re.search(r'(\d+)\s+estrellas', text)
    if match:
        return int(match.group(1))
    return None

def extractRecommendations(recommendations):
    # Splits a list of recommended dishes in the review, handling cases with "y" (e.g., "X y Y").

    recommendations_list = recommendations.split(', ')
    if ' y ' in recommendations_list[-1]:
        last_dishes = recommendations_list[-1].rsplit(' y ', 1)
        recommendations_list = recommendations_list[:-1] + last_dishes
    return recommendations_list

def convertToDate(date_text):
    # Converts relative date information (e.g., "2 weeks ago", "3 months ago") into an exact date.
    # It handles weeks, months, and years and returns the corresponding start date.

    today = datetime.today()

    if 'semana' in date_text:
        # Extract number of weeks, default to 1 if no number is present
        weeks = pd.Series(date_text).str.extract(r'(\d+)')[0]
        weeks = int(weeks.iloc[0]) if pd.notna(weeks.iloc[0]) else 1
        monday_of_current_week = today - timedelta(days=today.weekday())  # Get Monday of the current week
        return monday_of_current_week.date() - timedelta(weeks=weeks)

    elif 'mes' in date_text:
        # Extract number of months, default to 1 if no number is present
        months = pd.Series(date_text).str.extract(r'(\d+)')[0]
        months = int(months.iloc[0]) if pd.notna(months.iloc[0]) else 1
        target_date = today - relativedelta(months=months)
        # Return the first day of the target month
        return target_date.replace(day=1).date()

    elif 'año' in date_text:
        # Extract number of years, default to 1 if no number is present
        years = pd.Series(date_text).str.extract(r'(\d+)')[0]
        years = int(years.iloc[0]) if pd.notna(years.iloc[0]) else 1
        target_date = today - relativedelta(years=years)
        # Return the first day of the target year
        return target_date.replace(month=1, day=1).date()

    return None  # Return None if no match is found

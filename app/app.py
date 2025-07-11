import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys
import importlib

import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join('..')))
from src import plots
importlib.reload(plots)
from src import ml_processing
importlib.reload(ml_processing)
from src import llm_insights
importlib.reload(llm_insights)

import tab_1
importlib.reload(tab_1)
import tab_3
importlib.reload(tab_3)
import header
importlib.reload(header)

# Function to load data from uploaded file
@st.cache_data
def loadData(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Load all necessary files processed in sentiment.py
def loadAdditionalData(reviews, raw_path, processed_path):
    if 'embedding' in reviews.columns:
        # Convert embeddings from string to list of floats
        reviews['embedding'] = reviews['embedding'].apply(reFormatEmbeddings)

    file_name = uploaded_file.name
    place = extractPrefix(file_name)
    
    st.markdown(f"<h1 style='text-align: center; color: #000000;'>🍴 {place.upper()} 🍴</h1>", unsafe_allow_html=True)

    # Paths for the JSON and additional CSV files
    general_insights_file = os.path.join(processed_path, f"{place}_general_insights.json")
    worst_periods_file = os.path.join(processed_path, f"{place}_worst_periods_insights.json")
    sample_reviews_file = os.path.join(processed_path, f"{place}_sample_selected_reviews.csv")
    resume_file = os.path.join(raw_path, f"resumme_{place}.csv")
    
    # Load "place"_general_insights.json into a dictionary
    if os.path.exists(general_insights_file):
        general_insights = loadJson(general_insights_file)
        #st.write("General Insights:", general_insights)
    else:
        st.warning(f"{place}_general_insights.json not found in {processed_path}")

    # Load "place"_worst_periods_insights.json into a dictionary
    if os.path.exists(worst_periods_file):
        worst_periods_insights = loadJson(worst_periods_file)
        #st.write("Worst Periods Insights:", worst_periods_insights)
    else:
        st.warning(f"{place}_worst_periods_insights.json not found in {processed_path}")
    
    # Load "place"_sample_selected_reviews.csv into a DataFrame
    if os.path.exists(sample_reviews_file):
        sample_reviews = pd.read_csv(sample_reviews_file)
        #st.write("Sample Selected Reviews:")
        #st.dataframe(sample_reviews)
    else:
        st.warning(f"{place}_sample_selected_reviews.csv not found in {processed_path}")

    # Load resumme_"place".csv from ./data/raw into a DataFrame
    if os.path.exists(resume_file):
        resume = pd.read_csv(resume_file)
        #st.write(f"Resume data for {place}:")
        #st.dataframe(resume)
    else:
        st.warning(f"resumme_{place}.csv not found in {raw_path}")
    
    return place, reviews, sample_reviews, resume, general_insights, worst_periods_insights 

# Split the filename and extract the part before "_ml"
def extractPrefix(file_name):
    return file_name.split('_ml')[0]

# Load Json file with calculated insights
def loadJson(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Format correctly the Embeddings column
def reFormatEmbeddings(embedding_str):
    cleaned_str = re.sub(r'[\[\]\n]', '', embedding_str)
    embedding_list = list(map(float, cleaned_str.split()))
    return np.array(embedding_list, dtype=np.float32)
    return embedding_str

# Filter data for the last periods based on filter_min and filter_max
def addFilters(reviews, filter_min, filter_max):
    if filter_min is not None:
        filter_min = pd.to_datetime(filter_min)
    if filter_max is not None:
        filter_max = pd.to_datetime(filter_max)

    reviews['date'] = pd.to_datetime(reviews['date'])
    # Set default values for start_date and end_date
    limit_date = reviews['date'].max()
    start_date = filter_min if filter_min is not None else limit_date - pd.DateOffset(years=1)
    end_date = filter_max if filter_max is not None else limit_date

    # Apply filtering
    selected_reviews = reviews[(reviews['date'] >= start_date) & (reviews['date'] <= end_date)]
    return selected_reviews

# Format topics to write them in app
def format_topic_terms(terms):
    if isinstance(terms, list):
        return ", ".join([f'{weight}*"{term}"' for weight, term in terms])
    else:
        return str(terms) 
            
# Data Paths
processed_path = '../data/processed/'
raw_path = '../data/raw/'

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Reviews Dashboard",
    page_icon="🍽️",
    layout="wide",
)

# File uploader for CSV selection and all the necesary data
show_ml_lab_tab = st.sidebar.toggle(
    "Activate ML Lab Tab", 
    value=False, 
    help="Shows the ML processing details tab to adjust settings and see impact. Note: This may slow down dashboard performance."
)
enable_openai_api = st.sidebar.toggle(
    "Enable OpenAI API features", 
    value=False, 
    help="Enables or disables features that use the OpenAI API. Requires valid API keys and may incur additional costs."
)
st.sidebar.header("Select CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    ## Load all necessary data
    # Load reviews data and extract place from the file name
    reviews = loadData(uploaded_file)
    place, reviews, sample_reviews, resume, general_insights, worst_periods_insights = loadAdditionalData(reviews, raw_path, processed_path)

    # Label mapping for interest columns and label name
    label_mapping = {
        'rating_score': 'Rating',
        'food_score': 'Food',
        'service_score': 'Service',
        'atmosphere_score': 'Ambient'
    }
    label_keys = list(label_mapping.keys())
    additional_label_mapping = {
        'meal_type': 'Meal'
    }
    additional_label_keys = list(additional_label_mapping.keys())

    ## Header plots
    average_score = (resume['stars'] * resume['reviews']).sum() / resume['reviews'].sum()

    # Average score with stars
    stars = "⭐️" * int(round(average_score))
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; padding: 0px;">
            <h1 style="font-size: 50px; color: #4CAF50; margin: 0;">{round(average_score, 2)} {stars}</h1>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(f"<h4 style='text-align: center; color: #000000;'></h4>", unsafe_allow_html=True)

    col1, col2 = st.columns([10, 12])
    # Last 4 weeks trend
    with col1:
        
        fig_line = header.weekEvolution(reviews, label_mapping)
        st.markdown("<h4 style='text-align: left;'>📆 Last 4 weeks</h4>", unsafe_allow_html=True)
        st.plotly_chart(fig_line)

    # Distribution of reviews
    with col2:
        
        st.markdown("<h4 style='text-align: left;'>⭐ Distribution</h4>", unsafe_allow_html=True)
        col21, col22 = st.columns(2)
        # Donut chart for reviews distribution
        with col21:
            color_scale = ['#4CAF50', '#8BC34A', '#FFEB3B', '#FFC107', '#F44336']  # Green to Red scale
            resume['stars_label'] = resume['stars'].apply(lambda x: '⭐' * x)  # Convert stars to labels
            fig_donut = go.Figure(
                go.Pie(
                    labels=resume['stars_label'],
                    values=resume['reviews'],
                    hole=0.5,
                    marker=dict(colors=color_scale),
                    textinfo='percent+label',
                    insidetextorientation='radial'
                )
            )
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=20, b=50, l=80, r=80),
                height=250,
                width=250
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        
        # Bar chart for reviews count by score
        with col22:
            
            st.markdown("<h4 style='text-align: center;'> </h4>", unsafe_allow_html=True)
            fig_bar = go.Figure(
                go.Bar(
                    x=resume['stars_label'],
                    y=resume['reviews'],
                    marker=dict(color=color_scale),
                    text=resume['reviews'],
                    textposition='auto'
                )
            )
            fig_bar.update_xaxes(showgrid=False)
            fig_bar.update_yaxes(showgrid=False, showticklabels=False)
            fig_bar.update_layout(
                margin=dict(t=10, b=10, l=10, r=20),
                height=200,
                width=300,
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    ## Tabs
    if show_ml_lab_tab:
        tab1, tab2, tab3, tab4 = st.tabs([" 📋 Status ", " 📢 Customer Insigths ", " 🕵🏻‍♂️ Bad times Deep Dive ", " 🧪 ML Lab "])
    else:
        tab1, tab2, tab3 = st.tabs([" 📋 Status ", " 📢 Customer Insigths ", " 🕵🏻‍♂️ Bad times Deep Dive "])

    # Status tab
    with tab1:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'> 📋 Status 📋</h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        
        # Year Overview
        st.markdown("<h4 style='text-align: left; color: #00000;'>🗓️ Overview </h4>", unsafe_allow_html=True)
        st.write("Annual average rating trends over recent years show overall performance. Spot upward or downward shifts and compare average ratings in Food, Service, and Ambient to identify strengths and improvement areas.")
        reviews['year'] = reviews['date'].dt.year
        recent_reviews = reviews[reviews['date'] >= reviews['date'].max() - pd.DateOffset(years=8)]
        yearly_avg_scores = recent_reviews.groupby('year')[label_keys[0]].mean()

        fig = go.Figure(
            go.Scatter(
                x=yearly_avg_scores.index.astype(str),
                y=yearly_avg_scores.values,
                mode='lines+markers+text',  # Añade texto a los puntos
                line=dict(color='#32CD32', width=4),
                text=[f"{val:.2f}" for val in yearly_avg_scores],  # Valores como etiquetas
                textposition="top center",  # Posición de las etiquetas
                hoverinfo="text"
            )
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(t=20, b=20),
            height=280
        )

        col1, col2 = st.columns([6, 3])
        # Display yearly scroe
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display categories
        with col2:
            columns = list(label_mapping.keys())[1:]
            average_scores = [reviews[col].mean() for col in columns]
            colors = ['rgba(31, 119, 180, 0.8)', 'rgba(107, 174, 214, 0.8)', 'rgba(158, 202, 225, 0.8)']

            fig_categories = go.Figure(
                go.Bar(
                    x= np.round(average_scores, 2),
                    y=[label_mapping[col] for col in columns],  # Etiquetas usando label_mapping en orden
                    marker=dict(color=colors),
                    text=[f"{score:.2f}" for score in average_scores],
                    textposition='auto',
                    orientation='h',
                    name="Categories"
                )
            )
            fig_categories.update_layout(
                xaxis=dict(showgrid=False, range=[0, 5], tickvals=[0, 1, 2, 3, 4, 5]),
                yaxis=dict(showgrid=False),
                margin=dict(t=20, b=20),
                height=280
            )
            st.plotly_chart(fig_categories, use_container_width=True, key="fig_categories_tab1")

        # Trend Monthly Overview
        st.markdown("<h4 style='text-align: left ;'>📝 Monthly Overview</h4>", unsafe_allow_html=True)
        st.write("Provides a more detailed view of average scores for each category throughout the year. It highlights fluctuations over specific months, allowing stakeholders to observe seasonal variations or significant dips and peaks that may need further investigation.")
        recent_reviews = reviews[reviews['date'] >= reviews['date'].max() - pd.DateOffset(years=1)]
        fig_trend = tab_3.plotTrend(recent_reviews, label_mapping, app = True)
        st.plotly_chart(fig_trend, use_container_width=True, key="fig_month_trend_tab1")

        most_recommended, less_recommended = ml_processing.analyzeRecommendations(reviews)
        if len(most_recommended) > 0 and len(less_recommended) > 0:
            # Recommendations
            st.markdown("<h4 style='text-align: left; color: #000;'>🤩 Recommendations</h4>", unsafe_allow_html=True)
            st.write("Discover the top and least recommended items based on user feedback. Here are the highlights for the most popular and least favored choices from our reviews.")
            
            # CSS style for the list
            st.markdown(
                """
                <style>
                ul {
                    margin-top: 0;
                    margin-bottom: 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)
            # Most recommended
            with col1:
                st.markdown("<h5 style='text-align: left; color: #000;'>✚ Most Recommended</h5>", unsafe_allow_html=True)
                for item in most_recommended[:5]:
                    st.markdown(f"👌🏽 {item[0]} ({item[1]} times)")

            # Less recommended
            with col2:
                st.markdown("<h5 style='text-align: left; color: #000;'>− Least Recommended</h5>", unsafe_allow_html=True)
                for item in less_recommended[:5]:
                    st.markdown(f"🍽️ {item}")

        # Lasr reviews selection
        st.markdown("<h4 style='text-align: left; color: #00000;'>🚨 Last Reviews</h4>", unsafe_allow_html=True)
        st.write("Selection of recent reviews, separated into the best and worst ratings. Use this feedback to understand current strengths and pinpoint opportunities for improvement.")
        col1, col2 = st.columns(2)
        # recent_best_reviews
        with col1:
            recent_best_reviews = sample_reviews[sample_reviews['sample_type'] == 'recent_best_reviews'][['date', label_keys[0],'review', label_keys[1], label_keys[2], label_keys[3], additional_label_keys[0]]]
            recent_best_reviews.rename(columns = {'review':'Review', label_keys[0]:label_mapping[label_keys[0]], additional_label_keys[0]:additional_label_mapping[additional_label_keys[0]],label_keys[1]:label_mapping[label_keys[1]], label_keys[2]:label_mapping[label_keys[2]], label_keys[3]:label_mapping[label_keys[3]], 'date':'Date'}, inplace = True)
            st.markdown("<h5 style='text-align: left;'> 👍  Best!</h5>", unsafe_allow_html=True)
            st.dataframe(recent_best_reviews, height= 600)
        
        # recent_worst_reviews
        with col2:
            recent_worst_reviews = sample_reviews[sample_reviews['sample_type'] == 'recent_worst_reviews'][['date', label_keys[0],'review', label_keys[1], label_keys[2], label_keys[3], additional_label_keys[0]]]
            recent_worst_reviews.rename(columns = {'review':'Review', label_keys[0]:label_mapping[label_keys[0]], additional_label_keys[0]:additional_label_mapping[additional_label_keys[0]],label_keys[1]:label_mapping[label_keys[1]], label_keys[2]:label_mapping[label_keys[2]], label_keys[3]:label_mapping[label_keys[3]], 'date':'Date'}, inplace = True)
            st.markdown("<h5 style='text-align: left;'> 👎  Worst...</h5>", unsafe_allow_html=True)
            st.dataframe(recent_worst_reviews, height= 600)
        
    # Insights tab
    with tab2:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'> 📢 Customer Insights 📢 </h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.write("These insights summarize key themes and feedback extracted from all user reviews, highlighting strengths, pain points, and areas for improvement.")
        
        # General Insights
        col1, col2 = st.columns(2)
        # Strenghts
        with col1:
            st.markdown("<h5 style='text-align: center;'>💪 Strengths!</h5>", unsafe_allow_html=True)
            for insight in general_insights['best']:
                st.success('👍 ' + insight)

         # Pain points
        
        with col2:
            st.markdown("<h5 style='text-align: center;'>🤬 Pain Points...</h5>", unsafe_allow_html=True)
            for insight in general_insights['worst']:
                st.error('👎 ' + insight)

        _, col2, _ = st.columns([1, 3, 1])
        # Next steps
        with col2:
            st.markdown("<h4 style='text-align: center;'>💡 Areas for Improvement</h4>", unsafe_allow_html=True)
            for insight in general_insights['improve']:
                st.warning('⚠️ ' + insight)

        st.markdown("<h3 style='text-align: center; color: #00000;'></h3>", unsafe_allow_html=True)
        
        ## Filters
        col1, col2, col3 = st.columns([6, 2, 2])
        default_start_date = reviews['date'].max() - pd.DateOffset(years=1)
        default_end_date = reviews['date'].max()
        with col2:
            filter_min_tab2 = st.date_input("Start Date", default_start_date, key="filter_min_tab2")
        with col3:
            filter_max_tab2 = st.date_input("End Date", default_end_date, key="filter_max_tab2")
        # Apply the filter function
        sample_reviews_filtered = addFilters(sample_reviews, filter_min_tab2, filter_max_tab2)
        reviews_filtered = addFilters(reviews, filter_min_tab2, filter_max_tab2)

        # Sentiment status plot
        st.markdown("<h4 style='text-align: left; color: #00000;'> 💘 Sentiment</h4>", unsafe_allow_html=True)
        st.write("Monthly evolution of customer feelings over the past year, divided into positive, neutral, and negative categories. It highlights periods of higher satisfaction or concerns, helping to spot trends in customer sentiment.")
        fig = plots.plotSentimentTrend(reviews_filtered, years_limit = 2, app = True)
        st.plotly_chart(fig, use_container_width=True)

        if enable_openai_api:
            ## Dynamic insights
            st.markdown("<h4 style='text-align: left; color: #00000;'>📢 Customer insights for selected period</h4>", unsafe_allow_html=True)
            st.write("Summarize key themes and feedback extracted from user reviews of the selected period. Highlighting strengths, pain points, and areas for improvement.")

            _, col2 = st.columns([6, 2])
            # Adjust the number of insights extracted
            with col2:
                filter_number_insights = st.number_input("Number of Insights per category", min_value= 2, max_value=5, value=2, key="filter_number_insights")
                filter_number_insights = int(filter_number_insights) if filter_number_insights is not None else 2
            
            # Update topics with selected  reviews
            reviews_summary_dict = tab_1.updateTopicsDict(reviews_filtered)

            # Extract Insights with LLM
            client = llm_insights.initChatGPTClient()

            # Generate insights
            general_insights_prompt = (
                "I have this information extracted from LDA topics using clustering and sentiment analysis, including positive and negative terms, in JSON format.\n"
                f"I want you to extract:\n"
                f"- {filter_number_insights} positive points\n"
                f"- {filter_number_insights} negative points\n"
                f"- {filter_number_insights} improvement suggestions based on the negative points\n"
                "\n"
                "Each point should be a logical, simple, and concise sentence that provides value. Do not name specific terms or topics, but focus on delivering direct value to business stakeholders without ambiguity. If you mention something that didn't go well, give examples based on the information.\n"
                "Return the result in English in JSON format, ensuring it is easy to read in a notebook and standardized as follows:\n"
                "\n"
                "{best:['','',''], worst:['','',''], improve:['','','']}\n"
                "\n"
                "Ensure there are no contradictions between positive, negative, and improvement points.\n"
                "The information:\n"
            )
            insigths_summary_updated = llm_insights.extractInsightsWithLLM(reviews_summary_dict, general_insights_prompt, client)

            # Write insights
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h5 style='text-align: center;'>💪 Strengths!</h5>", unsafe_allow_html=True)
                for insight in insigths_summary_updated['best']:
                    st.success('👍 ' + insight)

            with col2:
                st.markdown("<h5 style='text-align: center;'>🤬 Pain Points...</h5>", unsafe_allow_html=True)
                for insight in insigths_summary_updated['worst']:
                    st.error('👎 ' + insight)

            _, col2, _ = st.columns([1, 3, 1])

            with col2:
                st.markdown("<h4 style='text-align: center;'>💡 Areas for Improvement</h4>", unsafe_allow_html=True)
                for insight in insigths_summary_updated['improve']:
                    st.warning('⚠️ ' + insight)

        # Extraction of best and worst reviews
        st.markdown("<h4 style='text-align: left; color: #00000;'>🤓 Reviews Overview</h4>", unsafe_allow_html=True)
        st.write("Here are recent high and low reviews, summarizing both positive feedback and areas for improvement from individual customers. Quick glance at what customers value most and where improvements can be made.")
        col1, col2 = st.columns(2)
        with col1:
            # best_reviews
            best_reviews = sample_reviews_filtered[sample_reviews_filtered['sample_type'] == 'best_reviews_sample'][['date', label_keys[0],'review', label_keys[1], label_keys[2], label_keys[3], additional_label_keys[0]]]
            best_reviews.rename(columns = {'review':'Review', label_keys[0]:label_mapping[label_keys[0]], additional_label_keys[0]:additional_label_mapping[additional_label_keys[0]],label_keys[1]:label_mapping[label_keys[1]], label_keys[2]:label_mapping[label_keys[2]], label_keys[3]:label_mapping[label_keys[3]], 'date':'Date'}, inplace = True)
            best_reviews.fillna('', inplace=True)
            st.markdown("<h5 style='text-align: left;'> 👍  Best!</h5>", unsafe_allow_html=True)
            st.dataframe(best_reviews, height=500)

        with col2:
            # worst_reviews
            worst_reviews = sample_reviews_filtered[sample_reviews_filtered['sample_type'] == 'worst_reviews_sample'][['date', label_keys[0],'review', label_keys[1], label_keys[2], label_keys[3], additional_label_keys[0]]]
            worst_reviews.rename(columns = {'review':'Review', label_keys[0]:label_mapping[label_keys[0]], additional_label_keys[0]:additional_label_mapping[additional_label_keys[0]],label_keys[1]:label_mapping[label_keys[1]], label_keys[2]:label_mapping[label_keys[2]], label_keys[3]:label_mapping[label_keys[3]], 'date':'Date'}, inplace = True)
            worst_reviews.fillna('', inplace=True) 
            st.markdown("<h5 style='text-align: left;'> 👎  Worst...</h5>", unsafe_allow_html=True)
            st.dataframe(worst_reviews, height=500)

    # Bad times tab
    with tab3:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'> 🕵🏻‍♂️ Bad Times Deep Dive 🕵🏻‍♂️ </h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.write("Identify the lowest-rated periods based on customer reviews, highlighting specific issues and improvement opportunities during times of lower satisfaction.")
        
        ## Filters
        col1, col2, col3 = st.columns([6, 2, 2])
        default_start_date = reviews['date'].max() - pd.DateOffset(years=1)
        default_end_date = reviews['date'].max()
        with col2:
            filter_min_tab3 = st.date_input("Start Date", default_start_date, key="filter_min_tab3")
        with col3:
            filter_max_tab3 = st.date_input("End Date", default_end_date, key="filter_max_tab3")

        # Apply the filter function
        reviews_filtered = addFilters(reviews, filter_min_tab3, filter_max_tab3)
        
        # Trend Overview
        st.markdown("<h4 style='text-align: left ;'>📝 Overview</h4>", unsafe_allow_html=True)
        st.write("Average monthly ratings across different categories. The chart helps pinpoint drops and peaks, providing context for periods with notable fluctuations in customer satisfaction.")
        fig = tab_3.plotTrend(reviews_filtered, label_mapping, app = True)
        st.plotly_chart(fig, use_container_width=True)

        # Problems by period
        st.markdown("<h4 style='text-align: left ;'>🔍 Period details</h4>", unsafe_allow_html=True)
        st.write("Customer feedback for the selected period, including specific problems and actionable improvement suggestions based on customer reviews. It allows a closer look at the challenges during low-rated periods.")

        # Filter low_score_periods based on filter_min and filter_max
        from datetime import datetime
        dates = list(worst_periods_insights.keys())
        dates = [datetime.strptime(date, '%Y-%m') for date in dates]
        limit_date = max(dates)

        start_date = pd.to_datetime(filter_min_tab3 if filter_min_tab3 is not None else (limit_date - pd.DateOffset(years=1)))
        end_date = pd.to_datetime(filter_max_tab3 if filter_max_tab3 is not None else limit_date)
        worst_periods_insights_filtered = {date: data for date, data in worst_periods_insights.items() if start_date <= datetime.strptime(date, '%Y-%m') <= end_date}
        
        # Display bad periods by month
        for i, (period, insights) in enumerate(sorted(worst_periods_insights_filtered.items(), key=lambda x: x[0], reverse=True)):
            expanded = True if i == 0 else False

            with st.expander(f"🗓️  {period}", expanded=expanded):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<h5 style='text-align: center;'>🤬 Problems</h5>", unsafe_allow_html=True)
                    for problem in insights['problems']:
                        st.error('💔 ' + problem)

                with col2:
                    st.markdown("<h5 style='text-align: center;'>🔧 Areas for Improvement</h5>", unsafe_allow_html=True)
                    for improvement in insights['improve']:
                        st.warning('💡' + improvement)

                # Reviews for the specific period
                period_reviews = sample_reviews[(sample_reviews['month'] == period) & (sample_reviews['sample_type'] == 'low_score_reviews')][['date', label_keys[0],'review', label_keys[1], label_keys[2], label_keys[3], additional_label_keys[0]]]
                period_reviews.rename(columns = {'review':'Review', label_keys[0]:label_mapping[label_keys[0]], additional_label_keys[0]:additional_label_mapping[additional_label_keys[0]],label_keys[1]:label_mapping[label_keys[1]], label_keys[2]:label_mapping[label_keys[2]], label_keys[3]:label_mapping[label_keys[3]], 'date':'Date'}, inplace = True)
                period_reviews.fillna('', inplace=True)
                if period_reviews.shape[0] > 0:
                    st.dataframe(period_reviews, height = 150)

    if show_ml_lab_tab:
    # ML Lab tab
        with tab4:
            st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #00000;'>🧪 ML Lab 🧪</h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
            st.write("""
            Do you have ML knowledge and want to play with algorithm parameters?  
            This advanced tab gives you a peek into the ML processes behind our insights. Here, you can experiment with:

            - **Sentence Communities:** See how reviews cluster around common themes.
            - **Dimensional Reduction (PCA & UMAP):** Adjust settings to visualize review patterns in scores or sentiment.
            - **k-Distance Graphs:** Find the best `eps` for DBSCAN clustering by tweaking distances.
            - **Clustering (DBSCAN with PCA & UMAP):** Explore how different settings group similar reviews.

            Dive in to understand how we extract meaningful insights from customer feedback!
            """)

            ## Filters
            col1, col2, col3 = st.columns([6, 2, 2])
            default_start_date = reviews['date'].max() - pd.DateOffset(years=1)
            default_end_date = reviews['date'].max()
            with col2:
                filter_min_ml = st.date_input("Start Date", default_start_date, key="filter_min_ml")
            with col3:
                filter_max_ml = st.date_input("End Date", default_end_date, key="filter_max_ml")

            # Apply the filter function
            reviews_filtered = addFilters(reviews, filter_min_ml, filter_max_ml)

            ## Plots
            # Communities
            st.markdown("<h4 style='text-align: left ;'>🫂 Sentence Communities</h4>", unsafe_allow_html=True)
            st.write("Shows how phrases in reviews group into communities based on meaning. By converting phrases into vectors, we can identify common themes, providing insights into recurring opinions about the venue.")
            print(reviews_filtered)
            if reviews_filtered.shape[0] > 200:
                plot_sample_reviews_filtered = reviews_filtered.sample(200).reset_index(drop=True)
            else:
                plot_sample_reviews_filtered = reviews_filtered.copy()
            fig = plots.plotCommunities(plot_sample_reviews_filtered, app = True)
            st.plotly_chart(fig, use_container_width=True)

            # Dimensional reduction and clustering
            st.markdown("<h4 style='text-align: left ;'>🧩 Dimensional reduction and clustering</h4>", unsafe_allow_html=True)
            st.write("To simplify analysis, we use dimensionality reduction techniques like PCA and UMAP. These help display complex data patterns in 2D, revealing trends related to review scores or sentiment.")
            st.write("")

            col1, col2 = st.columns(2)
            # PCA
            with col1:
                embeddings_pca, fig = ml_processing.calculateAndVisualizeEmbeddingsPCA(reviews_filtered, score_column = label_keys[0], plot = False, app = True)
                st.markdown("<h3 style='text-align: center ;'>⚙️ PCA</h3>", unsafe_allow_html=True)
                st.write("PCA projects reviews into a lower-dimensional space, retaining the most variance. Each point represents a review, colored by its rating. This visualization helps identify any clustering based on review scores.")
                st.plotly_chart(fig, use_container_width=True)
            
            # UMAP
            with col2:
                embeddings_umap, fig = ml_processing.calculateAndVisualizeEmbeddingsUMAP(reviews_filtered, plot = False, app = True)
                st.markdown("<h3 style='text-align: center ;'>⚙️ UMAP</h3>", unsafe_allow_html=True)
                st.write("UMAP preserves local structure, useful for detecting intricate patterns. Here, each point is a review, colored by sentiment. It shows if positive, neutral, and negative reviews form distinct groups.")
                st.plotly_chart(fig, use_container_width=True)
            
            # k-distance plots
            st.write("")
            st.write("To determine the optimal eps parameter for clustering the samples, we can use the **k-distance grap**, which allows us to find the optimal eps value. A strong increase indicates a suitable eps for well-defined clusters.")
            # k selection
            _, col2 = st.columns([10, 2])
            with col2:
                filter_k = st.number_input("Choose K", min_value=1, max_value=100, value=10, key="filter_k")
                filter_k = int(filter_k) if filter_k is not None else 10

            col1, col2 = st.columns(2)
            # k-distance PCA
            with col1:
                fig = plots.plotKdistance(embeddings_umap, k= filter_k, method='PCA', app = True)
                st.plotly_chart(fig, use_container_width=True)

            # k-distance UMAP
            with col2:
                fig = plots.plotKdistance(embeddings_pca, k= filter_k, method='UMAP', app = True)
                st.plotly_chart(fig, use_container_width=True)

            st.write("")
            st.write("Combining PCA or UMAP with DBSCAN helps identify clusters of similar reviews. Each point represents a review, colored by cluster, highlighting specific trends in customer feedback.")
            
            # eps and min samples selection
            _, col2, col3 = st.columns([4, 2, 2])
            with col2:
                filter_eps = st.number_input("Choose EPS value", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="filter_eps")
                filter_eps = float(filter_eps) if filter_eps is not None else 0.5
            with col3:
                filter_min_samples = st.number_input("Choose minimum samples", min_value=2, max_value=20, value=5, step=1, key="filter_min_samples")
                filter_min_samples = int(filter_min_samples) if filter_min_samples is not None else 5
            
            col1, col2 = st.columns(2)
            # dbscan PCA
            with col1:
                pca_clusters, fig = ml_processing.calculateAndVisualizeEmbeddingsPCA_with_DBSCAN(reviews_filtered, score_column = label_keys[0], eps=filter_eps, min_samples=filter_min_samples, plot = False, app = True)
                st.plotly_chart(fig, use_container_width=True)
        
            # dbscan UMAP
            with col2:
                umap_clusters, fig = ml_processing.calculateAndVisualizeEmbeddingsUMAP_with_DBSCAN(reviews_filtered, eps=filter_eps, min_samples=filter_min_samples, plot = False, app = True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Topics
            st.markdown("<h4 style='text-align: left ;'> 📚 Extract Topics </h4>", unsafe_allow_html=True)
            st.write("But what are the most important topics in each of our clusters? We can use the extraction of topics from the clusters to see which are the most important terms in our clusters. The terms of each topic have a weight assigned to them according to the information extracted. A higher weight means that the topic has a higher relevance than the rest of the terms in that topic, so we can consider it as one of the most important topics in the grouping of reviews.")

            # Refresh cluster columns with calculated in app
            if 'umap_cluster' in reviews_filtered.columns:
                reviews_filtered.drop(columns='umap_cluster', inplace=True)
            if 'pca_cluster' in reviews_filtered.columns:
                reviews_filtered.drop(columns='pca_cluster', inplace=True)

            reviews_filtered = (
                reviews_filtered
                .merge(pca_clusters[['review_id', 'pca_cluster']], on='review_id', how='left')
                .merge(umap_clusters[['review_id', 'umap_cluster']], on='review_id', how='left')
            )

            unique_pca_clusters = set(reviews_filtered['pca_cluster'].dropna().unique())
            unique_umap_clusters = set(reviews_filtered['umap_cluster'].dropna().unique())

            # Generate and display topics
            col1, col2 = st.columns(2)
            # PCA topics
            with col1:
                pca_topics = ml_processing.generateTopicsbyColumn(reviews_filtered, ['pca_cluster'])
                for cluster in unique_pca_clusters:
                    st.markdown(f"<h5 style='text-align: center;'>--- pca_cluster = {cluster} ---</h5>", unsafe_allow_html=True)
                    topics = pca_topics['pca_cluster'].get(cluster, None)
                    if topics:
                        for i, topic in enumerate(topics):
                            st.write(f"**Topic {i}:** {topic}")
                    else:
                        st.write("Insufficient number of group reviews to be able to calculate the topics, no topics generated for this group.")

            # UMAP topics
            with col2:
                umap_topics = ml_processing.generateTopicsbyColumn(reviews_filtered, ['umap_cluster'])
                for cluster in unique_umap_clusters:
                    st.markdown(f"<h5 style='text-align: center;'>--- umap_cluster = {cluster} ---</h5>", unsafe_allow_html=True)
                    topics = umap_topics['umap_cluster'].get(cluster, None)
                    
                    if topics:
                        for i, topic in enumerate(topics):
                            st.write(f"**Topic {i}:** {topic}")
                    else:
                        st.write("Insufficient number of group reviews to be able to calculate the topics, no topics generated for this group.")

else:
    st.write("Please upload a ML processed CSV file to start.")
import sys
import os
project_root = os.path.abspath("..")
sys.path.append(project_root)

import json
import pandas as pd
import numpy as np
import re

from openai import OpenAI
import openai_setup

# Create GPT Client
def initChatGPTClient():
    organization = openai_setup.conf['organization']
    project = openai_setup.conf['project']
    key = openai_setup.conf['key']

    client = OpenAI(
        api_key=key,
        organization=organization,
        project=project
    )
    return client

# Clean json outputs
def extract_json_string(input_string):
    json_match = re.search(r'\{.*\}', input_string, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(0)
        json_dict = json.loads(json_string)
        return json_dict
    return None

# Extract main insights from API
def extractInsightsWithLLM(info_dict, prompt, client):
    # Config and send message to gpt4o model
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a system expert in extracting value from reviews analysed using ML and NLP techniques, to provide valuable and actionable insights to stakeholders in an automated BI tool using AI."},
            {
                "role": "user",
                "content": prompt + str(info_dict)
            }
        ]
    )
    answer = completion.choices[0].message.content
    # Format correctly the answer
    answer_clean = extract_json_string(answer)
    return answer_clean
import requests
import os
from dotenv import load_dotenv
from os.path import join, dirname

load_dotenv()
HUGGINGFACE_API = os.environ.get("HUGGINGFACE_API")

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API}"}

def query_llm(payload):

	response = requests.post(API_URL, headers=headers, json=payload)

	return response.json()
	

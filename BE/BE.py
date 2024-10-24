from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from huggingface_hub import login
import nltk
import requests
from bs4 import BeautifulSoup
import re
import json
from flask_cors import CORS  # Import CORS from flask_cors


nltk.download('punkt')
nltk.download('punkt_tab')

model_name = "udinu/deberta-ner-furniture-products"

model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Define your label mapping
label_map = {
    "LABEL_0": "O",          # Outside of a named entity
    "LABEL_1": "B-product",  # Beginning of a product entity
    "LABEL_2": "I-product"   # Inside a product entity (if needed)
}

device = 0 if torch.cuda.is_available() else -1
# Specify device=0 to use the first GPU
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

def merge_tokens(ner_results):
  mapped_results = []
  current_result = {}

  for entity in ner_results:
      original_label = label_map.get(entity['entity'], entity['entity'])  # Default to original if not found
      word = entity['word']
      if original_label == 'B-product':
          if current_result:
              if not word.startswith("▁") and not word.startswith("_"):  # Check if the entity is not empty
                  current_result["word"] += word  # Continuation without space
              else:
                  current_result["word"] = current_result["word"][1:]
                  mapped_results.append(current_result)
                  current_result = {
                      'word': word,
                      'label': original_label,
                      'score': entity['score']
                  }
          else:
            current_result = {
                'word': word,
                'label': original_label,
                'score': entity['score']
            }

      elif original_label == 'I-product':
          if current_result:
            if not word.startswith("▁") and not word.startswith("_"):  # Check if the entity is not empty
              current_result["word"] += word  # Continuation without space
            else:
              current_result["word"] = current_result["word"][1:]
              mapped_results.append(current_result)
              current_result = {
                  'word': word,
                  'label': original_label,
                  'score': entity['score']
              }
          else:
            current_result = {
                'word': word,
                'label': original_label,
                'score': entity['score']
            }

  if current_result:
      current_result["word"] = current_result["word"][1:]
      mapped_results.append(current_result)

  return mapped_results

def merge_words(mapped_results):
  merged_products = []
  current_product = ""  # This will hold the current product being formed

  for entity in mapped_results:
      if entity['label'] == 'B-product':
          # If there's an existing product, finalize it before starting a new one
          if current_product:
              merged_products.append(current_product)
          # Start a new product
          current_product = entity['word']
      elif entity['label'] == 'I-product':
          if current_product:
              # Merge the current word into the ongoing product name
              current_product += " " + entity['word']  # Add with space
              # Update the minimum score if the current token's score is lower
          else:
              # If I-product appears without a B-product, you may choose to handle it (optional)
              continue

  # Finalize the last product if it exists
  if current_product:
      merged_products.append(current_product)

  return merged_products


nltk.download('punkt')  # Download necessary data for sentence tokenization
def fetch_page_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def split_product_list(text):
    # Use a regex to split on capitalized words and spaces
    product_list = re.split(r'(?<!\w)(?=\b[A-Z])', text)
    return [product.strip() for product in product_list if product.strip()]

def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    full_text = soup.get_text(separator=' ', strip=True)  # Get all text
    sentences = nltk.sent_tokenize(full_text)  # Split text into sentences
    return sentences



# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200", "methods": ["POST", "OPTIONS"]}})
stopwords = {"and", "for", "from", "give", "away"}


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "CORS preflight successful"}), 200
    
    print("POST request received at /predict")  # Debugging log

    data = request.json
    url = data.get("url", "")

    products = []
    results = []

    sentences = []
    content = fetch_page_content(url)
    # print(f"Extracted content from {url}: {content}")
    if content:
        sentences = extract_text(content)
    else:
        print(f"Failed to fetch content from {url}")

    for sentence in sentences:
        ner_results = ner_pipeline(sentence)
        results.append(ner_results)
        mapped_results = merge_tokens(ner_results)
        merged_products = merge_words(mapped_results)
        products.extend(merged_products)
    
    # print(sentences)
    # print(f"Extracted products from {url}: {products}")

    products = [product.capitalize() for product in products if len(product) > 2 and product.lower() not in stopwords]
    products = list(set(products))  # Remove duplicates
    products.sort()  # Optional: Sort for better readability
    response = jsonify({"products": products})
    response.headers['Content-Type'] = 'application/json'

    return jsonify({"products": products})



if __name__ == "__main__":
    app.run(debug=True)
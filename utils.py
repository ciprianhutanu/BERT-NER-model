from typing import List, Tuple, Dict
from bs4 import BeautifulSoup

import re
import time
import torch
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


def model_predict(text: str, batch_size_in_chars: int = 1500):
    paragraph = ''
    result = None

    words = text.split(' ')

    for i, word in enumerate(words):
        paragraph += word + ' '

        if i != len(words) - 1 and len(paragraph + words[i+1] + ' ') > batch_size_in_chars:
            products = extract_products(make_prediction(paragraph))
            if products is not None:
                if result is None:
                    result = set()

                result = result.union(products)
            paragraph = ''

    if paragraph:
        products = extract_products(make_prediction(paragraph))
        if products is not None:
            if result is None:
                result = set()

            result = result.union(products)

    if result is None:
        return None

    return list(result)


def make_prediction(paragraph: str):
    label_list = ['O', 'B-PRODUCT', 'I-PRODUCT']
    tokenizer = AutoTokenizer.from_pretrained('./product-recognition.model/')

    tokens = tokenizer(paragraph, padding=True)

    input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)

    model = AutoModelForTokenClassification.from_pretrained('./product-recognition.model/', num_labels=len(label_list))

    predictions = model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_labels = torch.argmax(predictions.logits.squeeze(), axis=1)
    predicted_labels = [label_list[i] for i in predicted_labels]

    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'])

    word_label_pairs = []
    current_word = ""
    current_label = ""
    for token, label in zip(decoded_tokens, predicted_labels):
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                word_label_pairs.append((current_word, current_label))
            current_word = token
            current_label = label

    if current_word:
        word_label_pairs.append((current_word, current_label))

    return word_label_pairs[1:-1]


def extract_products(word_label_pairs: List[Tuple[str, str]]):
    products = None
    current_product = ""

    for word, label in word_label_pairs:
        if label == "B-PRODUCT":
            if current_product and len(current_product.split()) > 1:
                if products is None:
                    products = set()

                products.add(current_product.strip())

            current_product = word.capitalize()

        elif label == "I-PRODUCT":

            if word.capitalize() not in current_product.split():
                current_product += " " + word.capitalize()

            else:
                if products is None:
                    products = set()
                if current_product and len(current_product.split()) > 1:
                    products.add(current_product.strip())

                current_product = word.capitalize()

        else:
            if current_product and len(current_product.split()) > 1:
                if products is None:
                    products = set()

                products.add(current_product.strip())

            current_product = ""

    if current_product and len(current_product.split()) > 1:
        if products is None:
            products = set()

        products.add(current_product.strip())

    return products


def urls_to_texts(urls: List[str], headers: Dict[str, str]):
    site_text = []
    status_codes = []
    sep = ['\n', '\t']

    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            raw_text = soup.get_text()

            site_clear_text = re.findall(r"\b[A-Za-z0-9().]+\b", raw_text)

            text = " ".join(site_clear_text)
            for s in sep:
                text = text.replace(s, "")

            site_text.append(text)
            status_codes.append(response.status_code)
        except requests.exceptions.RequestException as e:
            site_text.append("Unreachable Url")
            status_codes.append(500)

        time.sleep(1)

    return site_text, status_codes


def save_outputs(urls: List[str], status_codes: List[int], predictions: List[str]):
    df = pd.DataFrame({'url': urls, 'status_code': status_codes, 'prediction': predictions})

    table = pa.Table.from_pandas(df)

    pq.write_table(table, "output.parquet")


import os
import nltk
import numpy as np
import json
import random
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from openai import OpenAI

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Build vocabulary and training data
words = []
classes = []
documents = []
ignore_chars = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]))
classes = sorted(set(classes))


def bag_of_words(sentence):
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sentence)]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)


def predict_intent(sentence):
    bow = bag_of_words(sentence)
    model = tf.keras.models.load_model("chatbot_model.h5")
    result = model.predict(np.array([bow]))[0]
    threshold = 0.6
    results = [[i, r] for i, r in enumerate(result) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def extract_entities(sentence):
    doc = nlp(sentence)
    entities = {"names": [], "locations": [], "products": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["names"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            entities["locations"].append(ent.text)
        elif ent.label_ in ("PRODUCT", "ORG"):
            entities["products"].append(ent.text)
    return entities


def get_local_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return None


def get_openai_response(message, context=""):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": message}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't connect to the AI service. Error: {str(e)}"


def get_response(user_message):
    entities = extract_entities(user_message)
    intents_list = predict_intent(user_message)

    if intents_list:
        tag = intents_list[0]["intent"]
        confidence = float(intents_list[0]["probability"])
        if confidence > 0.75:
            response = get_local_response(tag)
            if entities["names"]:
                response = f"Hi {entities['names'][0]}! " + response
            return response

    # Fallback to OpenAI
    return get_openai_response(user_message)

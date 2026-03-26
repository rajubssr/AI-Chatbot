# AI Virtual Assistant Chatbot

An NLP-based virtual assistant built with Python, NLTK, TensorFlow, and Flask, with OpenAI API fallback for complex queries.

## Features
- Intent classification using a trained Neural Network (TensorFlow)
- Named Entity Recognition (NER) using spaCy — extracts names, locations, products
- OpenAI GPT fallback for queries outside trained intents
- Simple web interface using Flask

## Tech Stack
- Python, Flask
- NLTK, spaCy
- TensorFlow / Keras
- OpenAI API

## Project Structure
```
ai_chatbot/
├── app.py              # Flask web server
├── chatbot.py          # Core chatbot logic (NLP + OpenAI)
├── train.py            # Model training script
├── intents.json        # Intent patterns and responses
├── templates/
│   └── index.html      # Chat UI
└── requirements.txt
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Train the model (first time only)
python train.py

# Run the app
python app.py
```

Then open http://localhost:5000 in your browser.

## Model Details
- Architecture: Dense Neural Network (128 → 64 → output)
- Dropout: 0.5 (to prevent overfitting)
- Optimizer: Adam | Loss: Categorical Crossentropy
- Threshold: 0.75 confidence → falls back to OpenAI below threshold

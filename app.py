# Import necessary libraries
from flask import Flask, request, jsonify, render_template, session
import openai
import pandas as pd
import os
import spacy
import nltk
from nltk.corpus import stopwords
import logging
import time

# Initialize NLP tools
nltk.download('stopwords')
nltk.download('punkt')  # Needed for tokenization
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Configure logging to record interactions and performance metrics
logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure random key for session management

# Load OpenAI API key from a file
with open('keys.txt', 'r', encoding='utf-8') as arquivo:
    openai_key = arquivo.read().strip()
openai.api_key = openai_key

# Load client holdings and model portfolio data from CSV files
clients_df = pd.read_csv('financial_advisor_clients.csv')
models_df = pd.read_csv('client_target_allocations.csv')

# Verify that the data has been loaded correctly
print("Clients DataFrame columns:", clients_df.columns.tolist())
print("Models DataFrame columns:", models_df.columns.tolist())

# Preprocess data by filling missing values and ensuring correct data types
def preprocess_data():
    clients_df.fillna('', inplace=True)
    models_df.fillna('', inplace=True)
    clients_df['Client'] = clients_df['Client'].astype(str)
    models_df['Client'] = models_df['Client'].astype(str)

preprocess_data()

# Convert data to dictionaries for quick access based on 'Client' ID
client_holdings = clients_df.groupby('Client').apply(lambda x: x.to_dict('records')).to_dict()
model_portfolios = models_df.groupby('Client').apply(lambda x: x.to_dict('records')).to_dict()

# Initialize global counters for performance metrics
successful_interactions = 0
fallback_interactions = 0
error_count = 0

# Function to preprocess user input text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    doc = nlp(text)
    # Lemmatize tokens and remove stop words and punctuation
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_text = ' '.join(tokens)
    return processed_text

# Function to recognize the user's intent based on keywords
def recognize_intent(processed_text):
    # Define keyword sets for different intents
    holdings_keywords = {'holding', 'holdings', 'portfolio', 'investment', 'investments'}
    model_keywords = {'model', 'target', 'allocation', 'strategy'}
    strategy_keywords = {'strategy', 'recommend', 'recommendation', 'suggest', 'plan', 'advice'}

    tokens = set(processed_text.split())
    # Determine intent based on overlapping keywords
    if tokens & holdings_keywords:
        return 'get_holdings'
    elif tokens & model_keywords:
        return 'get_model_portfolio'
    elif tokens & strategy_keywords:
        return 'investment_strategy'
    else:
        return 'unknown'

# Function to extract entities (e.g., Client ID) from the user input
def extract_entities(text):
    doc = nlp(text)
    entities = {}
    # Extract entities recognized by spaCy (e.g., PERSON, ORG)
    for ent in doc.ents:
        if ent.label_ == 'PERSON' or ent.label_ == 'ORG':
            entities['Client'] = ent.text
    # Custom pattern matching for Client IDs like 'Client_6'
    client_id = None
    for token in doc:
        if token.text.lower().startswith('client_'):
            client_id = token.text
            break
    if client_id:
        entities['Client'] = client_id
    return entities

# Function to update the conversation context stored in the session
def update_context(entities):
    if 'conversation_context' not in session:
        session['conversation_context'] = {}
    session['conversation_context'].update(entities)
    logging.info(f"Updated Context: {session['conversation_context']}")

# Functions for logging performance metrics
def log_response_time(response_time):
    logging.info(f"Response Time: {response_time} seconds")

def log_intent(user_input, recognized_intent):
    logging.info(f"User Input: {user_input} | Recognized Intent: {recognized_intent}")

def log_entities(user_input, entities):
    logging.info(f"User Input: {user_input} | Extracted Entities: {entities}")

def increment_error_count():
    global error_count
    error_count += 1
    logging.info(f"Error Count: {error_count}")

def increment_successful_interactions():
    global successful_interactions
    successful_interactions += 1
    logging.info(f"Successful Interactions: {successful_interactions}")

def increment_fallback_interactions():
    global fallback_interactions
    fallback_interactions += 1
    logging.info(f"Fallback Interactions: {fallback_interactions}")

def calculate_session_metrics():
    session_end_time = time.time()
    session_duration = session_end_time - session.get('session_start_time', session_end_time)
    message_count = session.get('message_count', 0)
    logging.info(f"Session Duration: {session_duration} seconds")
    logging.info(f"Messages in Session: {message_count}")
    # Reset session metrics
    session.pop('session_start_time', None)
    session.pop('message_count', None)

# Flask before_request handler to initialize session metrics
@app.before_request
def before_request():
    if 'session_start_time' not in session:
        session['session_start_time'] = time.time()
        session['message_count'] = 0

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling chat messages
@app.route('/chat', methods=['POST'])
def chat():
    session['message_count'] += 1
    user_input = request.form.get('message')
    answer = generate_response(user_input)
    return jsonify({'response': answer})

# Route for resetting the conversation context
@app.route('/reset_context', methods=['POST'])
def reset_context():
    calculate_session_metrics()
    session.pop('conversation_context', None)
    return jsonify({'status': 'success'})

# Route for handling user feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    rating = data.get('rating')
    if rating:
        logging.info(f"User Feedback: {rating}")
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failure', 'reason': 'No rating provided'}), 400

# Functions to retrieve expected holdings and portfolio data for validation
def get_expected_holdings(client_id):
    holdings = client_holdings.get(client_id, [])
    holdings_text = '\n'.join([
        f"{item['Symbol']}: {item['Quantity']} shares"
        for item in holdings
        if 'Symbol' in item and 'Quantity' in item
    ]) or "No holdings information available."
    return holdings_text

def get_expected_portfolio(client_id):
    portfolio = model_portfolios.get(client_id, [])
    portfolio_text = '\n'.join([
        f"{item['Asset Class']}: {item['Target Allocation (%)']}% allocation"
        for item in portfolio
        if 'Asset Class' in item and 'Target Allocation (%)' in item
    ]) or "No model portfolio information available."
    return portfolio_text

# Function to generate the expected response based on the intent and client data
def generate_expected_response(intent, client_id):
    if intent == 'get_holdings':
        holdings_text = get_expected_holdings(client_id)
        expected_response = f"Client Holdings for {client_id}:\n{holdings_text}"
    elif intent == 'get_model_portfolio':
        portfolio_text = get_expected_portfolio(client_id)
        expected_response = f"Model Portfolio for {client_id}:\n{portfolio_text}"
    elif intent == 'investment_strategy':
        expected_response = None  # Skip strict validation for open-ended responses
    else:
        expected_response = "I'm sorry, I didn't understand your request."
    return expected_response

# Function to validate the bot's response against the expected response
def validate_response(bot_response, expected_response):
    if expected_response is None:
        # Skip validation for open-ended responses
        return True
    is_valid = expected_response.strip() in bot_response.strip()
    if not is_valid:
        logging.warning(f"Validation failed.\nExpected:\n{expected_response}\nBot Response:\n{bot_response}")
    return is_valid

# Main function to generate the chatbot's response
def generate_response(user_question):
    start_time = time.time()  # Start timing for response time metric
    processed_text = preprocess_text(user_question)
    intent = recognize_intent(processed_text)
    log_intent(user_question, intent)
    entities = extract_entities(user_question)
    log_entities(user_question, entities)
    update_context(entities)
    conversation_context = session.get('conversation_context', {})
    client_id = entities.get('Client') or conversation_context.get('Client')

    if not client_id:
        # If no client ID is found, prompt the user to provide one
        increment_fallback_interactions()
        response_text = "I'm sorry, I couldn't identify the Client in your question. Please specify a valid Client ID."
        log_interaction(user_question, intent, entities, response_text)
        end_time = time.time()
        response_time = end_time - start_time
        log_response_time(response_time)
        return response_text

    # Generate the expected response based on the intent and client data
    expected_response = generate_expected_response(intent, client_id)

    # Prepare data for the prompt to be sent to the OpenAI API
    holdings_text = get_expected_holdings(client_id)
    portfolio_text = get_expected_portfolio(client_id)

    # Construct the prompt based on the intent
    if intent == 'investment_strategy':
        prompt_content = f"""
You are an assistant helping financial advisors with client information.

Using only the data provided below, provide a recommended investment strategy for the client. Analyze the client's current holdings and model portfolio, and suggest any adjustments needed to align with their target allocations.

Client Holdings for {client_id}:
{holdings_text}

Model Portfolio for {client_id}:
{portfolio_text}

Answer:
"""
    else:
        # Prompt for other intents
        prompt_content = f"""
You are an assistant helping financial advisors with client information.

Only use the data provided below to answer the questions. Do not make up any information.

Advisor's Question: {user_question}

Client Holdings for {client_id}:
{holdings_text}

Model Portfolio for {client_id}:
{portfolio_text}

Answer:
"""

    # Prepare messages for the OpenAI ChatCompletion API
    messages = [
        {
            "role": "system",
            "content": "You are an assistant helping financial advisors with client information."
        },
        {
            "role": "user",
            "content": prompt_content
        }
    ]

    try:
        # Call the OpenAI API to generate the bot's response
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=250,
            temperature=0.7,
        )
        answer = response['choices'][0]['message']['content'].strip()
        increment_successful_interactions()
        log_interaction(user_question, intent, entities, answer)

        # Validate the bot's response
        is_valid = validate_response(answer, expected_response)
        if not is_valid:
            logging.warning(f"Validation failed for response: {answer}")
            # For open-ended responses, we can choose to keep the bot's response

        end_time = time.time()
        response_time = end_time - start_time
        log_response_time(response_time)
        return answer
    except Exception as e:
        # Handle any exceptions during the API call
        increment_error_count()
        logging.error(f"Error calling OpenAI API: {e}")
        response_text = "I'm sorry, I encountered an error while processing your request."
        log_interaction(user_question, intent, entities, response_text)
        end_time = time.time()
        response_time = end_time - start_time
        log_response_time(response_time)
        return response_text

# Function to log the interaction details
def log_interaction(user_input, intent, entities, response):
    logging.info(f"User Input: {user_input}")
    logging.info(f"Recognized Intent: {intent}")
    logging.info(f"Extracted Entities: {entities}")
    logging.info(f"Response: {response}")

# Run the Flask app
if __name__ == '__main__':
    if not os.path.isfile('financial_advisor_clients.csv') or not os.path.isfile('client_target_allocations.csv'):
        print("Data files not found. Please ensure 'financial_advisor_clients.csv' and 'client_target_allocations.csv' are in the same directory.")
        exit(1)
    app.run(debug=True)

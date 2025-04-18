import re
from datetime import datetime, timedelta
import json
from difflib import get_close_matches
import sqlite3
from typing import Dict, List, Optional, Tuple
import warnings
import nltk
import csv
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

warnings.filterwarnings('ignore')

class RobustTrainTicketChatbot:
    def __init__(self):
        # Conversation messages
        self.greetings = [
            "Hello! I'm your train ticket assistant. How can I help you today?",
            "Hi there! I can help you find the cheapest train tickets. What's your journey plan?",
            "Welcome to the train ticket chatbot! Where are you traveling to?"
        ]

        self.farewells = [
            "Thank you for using our service. Have a safe journey!",
            "Happy travels! Don't hesitate to come back if you need more help.",
            "Enjoy your trip! Come back anytime you need train tickets."
        ]

        # Conversation states
        self.STATES = {
            'GREETING': 0,
            'GET_DEPARTURE': 1,
            'GET_DESTINATION': 2,
            'GET_DATE': 3,
            'GET_TIME': 4,
            'GET_RETURN_DATE': 5,
            'GET_RETURN_TIME': 6,
            'CONFIRM_DETAILS': 7,
            'SEARCH_TICKETS': 8,
            'PRESENT_RESULTS': 9,
            'BOOKING': 10,
            'FAREWELL': 11
        }

        self.current_state = self.STATES['GREETING']
        self.conversation_state = {}
        #self.knowledge_base = self.load_knowledge_base('knowledge_base.csv')
        self.stations = self.load_stations('D:/Documents/Uni Stuff/Data Science/AI/CW 2/stations.csv')
        self.initialize_nlp_components()
        self.db_conn = sqlite3.connect('chatbot_db.sqlite', check_same_thread=False)
        self.create_tables()

    def load_knowledge_base(self, path: str) -> Dict[str, str]:
        knowledge = {}
        try:
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    question = row.get('question') or ''
                    answer = row.get('answer') or ''
                    knowledge[question.lower().strip()] = answer.strip()
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
        return knowledge

    def load_stations(self, path: str) -> List[str]:
        stations = []
        try:
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        stations.append(row[0].strip())
        except Exception as e:
            print(f"Error loading stations: {e}")
        return stations

    def initialize_nlp_components(self):
        try:
            try:
                import spacy
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                    print("spaCy NLP model loaded successfully.")
                except:
                    self.nlp = None
                    print("spaCy model not found. Some advanced features will be disabled.")
            except ImportError:
                self.nlp = None
                print("spaCy not installed. Some advanced features will be disabled.")

            try:
                from nltk.stem import PorterStemmer, WordNetLemmatizer
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
                print("NLTK components initialized successfully.")
            except ImportError:
                self.stemmer = None
                self.lemmatizer = None
                print("NLTK not installed. Some features will be disabled.")

            self.nlp_available = self.nlp is not None or (self.stemmer and self.lemmatizer)

        except Exception as e:
            print(f"Warning: NLP initialization encountered an error - {str(e)}")
            self.nlp_available = False

    def basic_text_processing(self, text: str) -> Dict:
        tokens = text.split()
        return {
            'tokens': tokens,
            'pos_tags': [(token, 'UNK') for token in tokens],
            'lemmas': tokens,
            'stems': tokens,
            'entities': []
        }

    def advanced_nlp_processing(self, text: str) -> Dict:
        result = {
            'tokens': [],
            'pos_tags': [],
            'lemmas': [],
            'stems': [],
            'entities': []
        }

        try:
            tokens = nltk.word_tokenize(text)
            result['tokens'] = tokens
            result['pos_tags'] = nltk.pos_tag(tokens)

            if self.lemmatizer:
                result['lemmas'] = [self.lemmatizer.lemmatize(token) for token in tokens]
            else:
                result['lemmas'] = tokens

            if self.stemmer:
                result['stems'] = [self.stemmer.stem(token) for token in tokens]
            else:
                result['stems'] = tokens

            if self.nlp:
                doc = self.nlp(text)
                result['entities'] = [(ent.text, ent.label_) for ent in doc.ents]

            return result if any(result.values()) else self.basic_text_processing(text)

        except Exception as e:
            print(f"NLP processing error: {str(e)}")
            return self.basic_text_processing(text)

    def log_conversation(self, user_id: str, message_type: str, content: str):
        try:
            nlp_results = self.advanced_nlp_processing(content)

            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (
                    user_id, timestamp, message_type, content,
                    tokens, pos_tags, lemmas, stems, entities
                )
                VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                message_type,
                content,
                json.dumps(nlp_results['tokens']),
                json.dumps(nlp_results['pos_tags']),
                json.dumps(nlp_results['lemmas']),
                json.dumps(nlp_results['stems']),
                json.dumps(nlp_results['entities'])
            ))
            self.db_conn.commit()
        except Exception as e:
            print(f"Failed to log conversation: {str(e)}")

    def create_tables(self):
        try:
            cursor = self.db_conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp DATETIME,
                    message_type TEXT,
                    content TEXT,
                    tokens TEXT,
                    pos_tags TEXT,
                    lemmas TEXT,
                    stems TEXT,
                    entities TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_departure_station TEXT,
                    preferred_destination_station TEXT,
                    travel_times TEXT
                )
            ''')

            self.db_conn.commit()
        except Exception as e:
            print(f"Database error: {str(e)}")

    def generate_response(self, user_input: str, user_id: str = "default") -> str:
        try:
            self.log_conversation(user_id, "user", user_input)

            if self.current_state == self.STATES['GREETING']:
                self.conversation_state['user_id'] = user_id
                self.current_state = self.STATES['GET_DEPARTURE']
                return self.greetings[0]

            if any(word in user_input.lower() for word in ['bye', 'goodbye', 'exit', 'quit']):
                self.current_state = self.STATES['FAREWELL']
                return self.farewells[0]

            return "I'm sorry, I didn't understand that. Could you please rephrase?"

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error. Please try again."
    def chat(self, message):
        # For now, just echo back
        return f"You said: {message}"


if __name__ == "__main__":
    print("Initializing robust chatbot...")
    chatbot = RobustTrainTicketChatbot()

    # print("\nChatbot:", chatbot.generate_response("Hello"))

    # print("Welcome to TrainBot! Type 'exit' to quit.")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == 'exit':
    #         print("TrainBot: Goodbye!")
    #         break
    #     response = chatbot.chat(user_input)
    #     print("TrainBot:", response)

    test_phrases = [
        "I want to travel from London to Cambridge",
        "on March 15th",
        "in the morning",
        "yes that's correct"
    ]

    for phrase in test_phrases:
        print("\nUser:", phrase)
        print("Chatbot:", chatbot.generate_response(phrase))

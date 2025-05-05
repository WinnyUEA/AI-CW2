"""
Train Ticket Chatbot with Delay Prediction
Simplified version without machine learning dependencies
"""

import re
import sqlite3
import random
from typing import Dict, List, Optional
import spacy
from spacy.matcher import Matcher

# Initialize spaCy for NLP
try:
    nlp = spacy.load('en_core_web_sm')
    print("NLP model loaded successfully.")
except:
    print("spaCy not available. Some features will be limited.")
    nlp = None

class RobustTrainTicketChatbot:
    def __init__(self):
        """Initialize the chatbot with all required components"""
        # Conversation messages
        self.greetings = ["Hello! How can I help with your train journey today?"]
        self.farewells = ["Thank you for using our service. Safe travels!"]

        # Conversation states
        self.STATES = {
            'GREETING': 0,
            'GET_DEPARTURE': 1,
            'GET_DESTINATION': 2,
            'DELAY_INQUIRY': 3,
            'GET_TRAIN_INFO': 4,
            'GET_CURRENT_STATION': 5,
            'GET_DELAY_DURATION': 6,
            'PREDICT_ARRIVAL': 7,
            'FAREWELL': 8
        }
        
        self.current_state = self.STATES['GREETING']
        self.conversation_state = {}
        
        # Initialize stations and schedules
        self.stations = [
            "Norwich", "Diss", "Stowmarket", "Ipswich",
            "Manningtree", "Colchester", "Chelmsford",
            "Stratford", "London Liverpool Street"
        ]
        
        self.initialize_train_schedules()
        
        # Set up NLP
        if nlp:
            self.matcher = Matcher(nlp.vocab)
            self.initialize_nlp_matchers()
        
        # Initialize database
        self.db_conn = sqlite3.connect(':memory:')  # Using in-memory database for simplicity
        self.create_tables()

    def create_tables(self):
        """Create database tables for conversation logging"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.db_conn.commit()

    def initialize_train_schedules(self):
        """Initialize train schedule data"""
        self.train_schedules = {
            "Norwich-LST": {
                "stations": self.stations,
                "times": [0, 25, 40, 50, 65, 80, 100, 115, 120]  # Minutes from origin
            },
            "LST-Norwich": {
                "stations": list(reversed(self.stations)),
                "times": [0, 5, 25, 45, 60, 70, 85, 100, 120]
            }
        }

    def initialize_nlp_matchers(self):
        """Set up NLP patterns for travel phrases"""
        # Pattern for "from <station>"
        self.matcher.add("DEPARTURE", [
            [{"LOWER": {"IN": ["from", "departing"]}}, {"ENT_TYPE": "GPE"}]
        ])
        # Pattern for "to <station>"
        self.matcher.add("DESTINATION", [
            [{"LOWER": {"IN": ["to", "going"]}}, {"ENT_TYPE": "GPE"}]
        ])

    def match_station(self, text: str) -> Optional[str]:
        """Match user input to known station names"""
        text = text.lower()
        for station in self.stations:
            if text in station.lower():
                return station
        return None

    def predict_arrival(self, route: str, current_station: str, delay: int) -> Optional[int]:
        """Predict arrival time at destination with delay"""
        try:
            schedule = self.train_schedules[route]
            current_idx = schedule["stations"].index(current_station)
            dest_idx = schedule["stations"].index("London Liverpool Street")
            remaining_time = schedule["times"][dest_idx] - schedule["times"][current_idx]
            return remaining_time + delay
        except:
            return None

    def log_conversation(self, user_input: str, response: str):
        """Log conversation to database"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_input, bot_response)
            VALUES (?, ?)
        ''', (user_input, response))
        self.db_conn.commit()

    def generate_response(self, user_input: str) -> str:
        """Generate appropriate response based on current state"""
        # Log the conversation
        self.log_conversation(user_input, "")
        
        # Detect intent
        intent = self.detect_intent(user_input)
        
        if intent == "farewell":
            self.current_state = self.STATES['FAREWELL']
            return random.choice(self.farewells)
        elif intent == "delay_inquiry":
            self.current_state = self.STATES['DELAY_INQUIRY']
            return "I can help with delay predictions. Which train are you on? (Norwich to London or London to Norwich)"
        
        # State machine logic
        if self.current_state == self.STATES['GREETING']:
            self.current_state = self.STATES['GET_DEPARTURE']
            return random.choice(self.greetings)
        
        elif self.current_state == self.STATES['GET_DEPARTURE']:
            station = self.match_station(user_input)
            if station:
                self.conversation_state['departure'] = station
                self.current_state = self.STATES['GET_DESTINATION']
                return f"Departing from {station}. Where to?"
            return "Station not recognized. Please try again."
        
        elif self.current_state == self.STATES['GET_DESTINATION']:
            station = self.match_station(user_input)
            if station:
                self.conversation_state['destination'] = station
                return f"Great! From {self.conversation_state['departure']} to {station}. How else can I help?"
            return "Station not recognized. Please try again."
        
        elif self.current_state == self.STATES['DELAY_INQUIRY']:
            if "norwich" in user_input.lower() and "london" in user_input.lower():
                self.conversation_state['route'] = "Norwich-LST"
            elif "london" in user_input.lower() and "norwich" in user_input.lower():
                self.conversation_state['route'] = "LST-Norwich"
            else:
                return "Please specify 'Norwich to London' or 'London to Norwich'"
            self.current_state = self.STATES['GET_CURRENT_STATION']
            return "At which station is the train currently?"
        
        elif self.current_state == self.STATES['GET_CURRENT_STATION']:
            station = self.match_station(user_input)
            if station:
                self.conversation_state['current_station'] = station
                self.current_state = self.STATES['GET_DELAY_DURATION']
                return "How many minutes has the train been delayed?"
            return "Station not recognized. Please try again."
        
        elif self.current_state == self.STATES['GET_DELAY_DURATION']:
            match = re.search(r'(\d+)', user_input)
            if match:
                delay = int(match.group(1))
                prediction = self.predict_arrival(
                    self.conversation_state['route'],
                    self.conversation_state['current_station'],
                    delay
                )
                if prediction:
                    return f"With {delay} minute delay, your train should arrive in London in about {prediction} minutes."
                return "Couldn't calculate arrival time. Please check your information."
            return "Please specify the delay in minutes (e.g., '15 minutes')."
        
        return "I'm not sure what you mean. Could you rephrase?"

    def detect_intent(self, text: str) -> str:
        """Detect user intent from input text"""
        text = text.lower()
        if any(word in text for word in ['hello', 'hi']):
            return "greeting"
        if any(word in text for word in ['bye', 'goodbye']):
            return "farewell"
        if any(word in text for word in ['delay', 'late', 'arrival']):
            return "delay_inquiry"
        return "unknown"


if __name__ == "__main__":
    print("Initializing Train Ticket Chatbot...")
    chatbot = RobustTrainTicketChatbot()

    print("\n🚆 Welcome to TrainBot! Type 'exit' to quit.")
    print("You can ask about:")
    print("- Tickets (e.g., 'I want to go from Norwich to London')")
    print("- Delays (e.g., 'My train is delayed')")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("TrainBot: Goodbye!")
            break
        response = chatbot.generate_response(user_input)
        print("TrainBot:", response)
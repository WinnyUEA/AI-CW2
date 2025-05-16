import re
import sqlite3
import random
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from typing import Dict, List, Optional, Tuple
import spacy
from spacy.matcher import Matcher
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlencode
from datetime import datetime, timedelta
import time

# Initialize spaCy for NLP
try:
    nlp = spacy.load('en_core_web_sm')
    print("NLP model loaded successfully.")
except:
    print("spaCy not available. Some features will be limited.")
    nlp = None

class TrainTicketChatbotWithBooking:
    def __init__(self):
        """Initialize the chatbot with all required components"""
        # Conversation messages
                # Initialize NLP processors
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Update your existing initialization
        self.initialize_nlp_components()
        
        self.greetings = [
            "Hello! I'm TrainBot, how can I help with your train journey today?",
            "Hi there! How can I assist with your train travel?"
        ]
        self.farewells = ["Thank you for using our service. Safe travels!"]
        self.options_prompt = "What would you like to do today? You can:\n1. Book tickets\n2. Check for delays\n3. Exit"

        # Conversation states
        self.STATES = {
            'GREETING': 0,
            'GET_OPTION': 1,
            'PARSE_BOOKING': 2,
            'GET_MISSING_INFO': 3,
            'CONFIRM_BOOKING': 4,
            'FAREWELL': 5,
            'DELAY_INQUIRY': 6,
            'GET_CURRENT_STATION': 7,
            'GET_DELAY_DURATION': 8
        }
        
        self.current_state = self.STATES['GREETING']
        self.booking_details = {
            'origin': None,
            'destination': None,
            'date': None,
            'time': None,
            'adults': '1',
            'children': '0',
            'is_return': False,
            'return_date': None,
            'return_time': None
        }
        self.missing_fields = []
        
        # Load station data from CSV
        self.load_station_data('stations.csv')  # Update with your CSV path
        
        # Initialize train schedules for delay prediction
        self.initialize_train_schedules()
        
        # Set up NLP
        if nlp:
            self.matcher = Matcher(nlp.vocab)
            self.initialize_nlp_matchers()
        
        # Initialize database
        self.db_conn = sqlite3.connect(':memory:')  # Using in-memory database for simplicity
        self.create_tables()

        # Load and process delay data
        self.load_delay_data()

    def load_delay_data(self):
        """Load and process delay data from CSV files"""
        self.delay_data = {
            'Norwich-LST': [],
            'LST-Norwich': []
        }
        
        # Load data for Norwich to London route
        for year in [2022, 2023, 2024]:
            try:
                df = pd.read_csv(f'{year}_service_details_Norwich_to_London.csv')
                df['route'] = 'Norwich-LST'
                self.delay_data['Norwich-LST'].append(df)
            except FileNotFoundError:
                print(f"Warning: {year}_service_details_Norwich_to_London.csv not found")
        
        # Load data for London to Norwich route
        for year in [2022, 2023, 2024]:
            try:
                df = pd.read_csv(f'{year}_service_details_London_to_Norwich.csv')
                df['route'] = 'LST-Norwich'
                self.delay_data['LST-Norwich'].append(df)
            except FileNotFoundError:
                print(f"Warning: {year}_service_details_London_to_Norwich.csv not found")
        
        # Combine all data for each route
        for route in self.delay_data:
            if self.delay_data[route]:
                self.delay_data[route] = pd.concat(self.delay_data[route])
                print(f"Loaded {len(self.delay_data[route])} records for {route} route")
        
        # Extract station sequences from the data
        self.extract_station_sequences()
        self.preprocess_delay_data()

    def extract_station_sequences(self):
        """Extract station sequences from the delay data"""
        self.station_sequences = {}
        
        for route, df in self.delay_data.items():
            if isinstance(df, pd.DataFrame):
                # Get unique stations in order of appearance for each train (rid)
                station_order = df.groupby('rid').apply(
                    lambda x: x.sort_values('location')['location'].tolist()
                )
                
                # Find the most common sequence
                if not station_order.empty:
                    most_common = max(set(tuple(seq) for seq in station_order), 
                                    key=lambda x: list(station_order).count(x))
                    self.station_sequences[route] = list(most_common)
                    print(f"Station sequence for {route}: {self.station_sequences[route]}")

    def preprocess_delay_data(self):
        """Clean and prepare delay data for analysis"""
        for route in self.delay_data:
            if isinstance(self.delay_data[route], pd.DataFrame):
                df = self.delay_data[route]
                
                # Print column names for debugging
                print(f"\nColumns in {route} data:")
                print(df.columns.tolist())
                
                # Handle each route's specific column names
                if route == 'Norwich-LST':
                    # Rename columns to standard names
                    df = df.rename(columns={
                        'planned_arrival_time': 'planned_arrival',
                        'planned_departure_time': 'planned_departure',
                        'actual_arrival_time': 'actual_arrival',
                        'actual_departure_time': 'actual_departure'
                    })
                    
                    # Remove duplicate column if exists
                    if 'late_canc_reason.1' in df.columns:
                        df = df.drop(columns=['late_canc_reason.1'])
                        
                elif route == 'LST-Norwich':
                    # Rename columns to standard names
                    df = df.rename(columns={
                        'gbtt_pta': 'planned_arrival',
                        'gbtt_ptd': 'planned_departure',
                        'actual_ta': 'actual_arrival',
                        'actual_td': 'actual_departure'
                    })
                
                # Convert time columns to datetime.time
                time_cols = ['planned_arrival', 'planned_departure', 'actual_arrival', 'actual_departure']
                for col in time_cols:
                    if col in df:
                        # Handle empty strings by replacing them with NaNs first
                        df[col] = df[col].replace('', pd.NA)
                        try:
                            df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce').dt.time
                        except:
                            # Try different format if %H:%M fails
                            df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time
                
                # Calculate delay minutes (arrival delay)
                df['delay_minutes'] = 0
                mask = df['actual_arrival'].notna() & df['planned_arrival'].notna()
                
                # Calculate delay in minutes
                df.loc[mask, 'delay_minutes'] = df[mask].apply(
                    lambda x: (datetime.combine(datetime.today(), x['actual_arrival']) - 
                            datetime.combine(datetime.today(), x['planned_arrival'])).total_seconds() / 60,
                    axis=1
                )
                
                # Filter out extreme delays (more than 120 minutes)
                df = df[df['delay_minutes'] <= 120]
                
                # Update the dataframe in delay_data
                self.delay_data[route] = df
                
                # Calculate typical delays between stations
                self.calculate_typical_delays(route, df)

    def calculate_typical_delays(self, route, df):
        """Calculate typical delays between stations for a route"""
        if route not in self.station_sequences:
            return
            
        stations = self.station_sequences[route]
        delay_stats = {}
        
        for i in range(len(stations)-1):
            from_stn = stations[i]
            to_stn = stations[i+1]
            pair = (from_stn, to_stn)
            
            # Get all services that stopped at both stations
            from_df = df[df['location'] == from_stn]
            to_df = df[df['location'] == to_stn]
            
            # Merge on rid and date_of_service to match the same train
            merged = pd.merge(
                from_df[['rid', 'date_of_service', 'actual_departure', 'delay_minutes']], 
                to_df[['rid', 'date_of_service', 'actual_arrival', 'delay_minutes']], 
                on=['rid', 'date_of_service'],
                suffixes=('_from', '_to')
            )
            
            if not merged.empty:
                # Calculate additional delay between stations
                merged['additional_delay'] = merged['delay_minutes_to'] - merged['delay_minutes_from']
                
                # Store statistics
                delay_stats[pair] = {
                    'mean_delay': merged['additional_delay'].mean(),
                    'median_delay': merged['additional_delay'].median(),
                    'std_dev': merged['additional_delay'].std(),
                    'count': len(merged)
                }
        
        self.delay_stats = delay_stats
        print(f"Delay statistics calculated for {route}:")
        for pair, stats in delay_stats.items():
            print(f"{pair[0]}->{pair[1]}: mean={stats['mean_delay']:.1f} min, median={stats['median_delay']:.1f} min")
    def load_station_data(self, csv_path: str):
        """Load station data from CSV file with robust error handling"""
        # Default stations that must be available
        self.essential_stations = {
            'London Liverpool Street': 'LST',
            'Norwich': 'NRW',
            'Diss': 'DIS',
            'Ipswich': 'IPS'
        }
        
        self.stations = list(self.essential_stations.keys())
        self.station_codes = {k.lower(): v for k, v in self.essential_stations.items()}
        
        # Add common abbreviations
        self.station_codes.update({
            'london': 'LST',
            'ldn': 'LST',
            'liverpool st': 'LST',
            'liv st': 'LST',
            'liverpool street': 'LST'
        })
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV with {len(df)} rows")
            
            for _, row in df.iterrows():
                try:
                    station_name = str(row['name']).strip()
                    tiploc = str(row.get('alpha3', '')).strip()
                    
                    if station_name and station_name != '\\N' and tiploc:
                        if station_name not in self.stations:
                            self.stations.append(station_name)
                        self.station_codes[station_name.lower()] = tiploc
                        
                        # Handle aliases if column exists
                        if 'longname.name_alias' in row:
                            alias = str(row['longname.name_alias']).strip()
                            if alias and alias != '\\N':
                                self.station_codes[alias.lower()] = tiploc
                except Exception as e:
                    print(f"Error processing row {_}: {e}")
                    continue
                    
            print(f"Loaded {len(self.stations)} stations total")
            print(f"Sample stations: {self.stations[:5]}")
        except Exception as e:
            print(f"Error loading CSV, using essential stations only: {e}")

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

    def log_conversation(self, user_input: str, response: str):
        """Log conversation to database"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_input, bot_response)
            VALUES (?, ?)
        ''', (user_input, response))
        self.db_conn.commit()

    def initialize_train_schedules(self):
        """Initialize train schedule data"""
        self.train_schedules = {
            "Norwich-LST": {
                "stations": ['Norwich', 'Diss', 'Ipswich', 'London Liverpool Street'],
                "times": [0, 25, 40, 120]  # Minutes from origin
            },
            "LST-Norwich": {
                "stations": ['London Liverpool Street', 'Ipswich', 'Diss', 'Norwich'],
                "times": [0, 25, 45, 120]
            }
        }

    def initialize_nlp_matchers(self):
        """Set up NLP patterns for travel phrases"""
        if nlp:
            # Pattern for "from <station>"
            self.matcher.add("DEPARTURE", [
                [{"LOWER": {"IN": ["from", "departing"]}}, {"ENT_TYPE": "GPE"}]
            ])
            # Pattern for "to <station>"
            self.matcher.add("DESTINATION", [
                [{"LOWER": {"IN": ["to", "going"]}}, {"ENT_TYPE": "GPE"}]
            ])

    def match_station(self, text: str) -> Optional[str]:
        """Robust station matching with multiple fallback strategies"""
        if not text:
            return None
            
        text = text.lower().strip()
        print(f"\nMatching station for: '{text}'")  # Debug
        
        # 1. Check direct mappings first
        direct_mappings = {
            'london': 'London Liverpool Street',
            'ldn': 'London Liverpool Street',
            'norwich': 'Norwich',
            'diss': 'Diss',
            'ipswich': 'Ipswich',
            'liverpool st': 'London Liverpool Street',
            'liv st': 'London Liverpool Street',
            'liverpool street': 'London Liverpool Street'
        }
        
        if text in direct_mappings:
            print(f"Matched via direct mapping: {direct_mappings[text]}")
            return direct_mappings[text]
        
        # 2. Check station codes
        for station_name, code in self.essential_stations.items():
            if text == code.lower():
                print(f"Matched station code: {station_name}")
                return station_name
        
        # 3. Check exact matches
        for station in self.stations:
            if text == station.lower():
                print(f"Exact match found: {station}")
                return station
                
        # 4. Check partial matches
        for station in self.stations:
            if text in station.lower():
                print(f"Partial match found: {station}")
                return station
                
        # 5. Try to extract from phrases like "london station"
        if 'station' in text:
            base_name = text.replace('station', '').strip()
            for station in self.stations:
                if base_name in station.lower():
                    print(f"Matched station from phrase: {station}")
                    return station
        
        print(f"No match found for: '{text}'")
        return None

    def get_station_code(self, station_name: str) -> Optional[str]:
        """Get station code from station name"""
        return self.station_codes.get(station_name.lower())

    def get_station_name(self, code: str) -> Optional[str]:
        """Get station name from station code"""
        for station, station_code in self.station_codes.items():
            if station_code == code:
                return station
        return None

    def parse_relative_date(self, date_str: str) -> Optional[str]:
        """Parse relative date expressions like 'today', 'tomorrow', 'next week' etc."""
        today = datetime.now().date()
        date_str = date_str.lower()
        
        if date_str in ['today', 'now']:
            return today.strftime('%d/%m/%Y')
        elif date_str == 'tomorrow':
            return (today + timedelta(days=1)).strftime('%d/%m/%Y')
        elif date_str == 'day after tomorrow':
            return (today + timedelta(days=2)).strftime('%d/%m/%Y')
        elif date_str.startswith('in ') or date_str.startswith('next '):
            # Handle expressions like "in 3 days", "next week"
            parts = date_str.split()
            if len(parts) == 2 and parts[1] == 'week':
                return (today + timedelta(weeks=1)).strftime('%d/%m/%Y')
            elif len(parts) == 3 and parts[1].isdigit():
                num = int(parts[1])
                if parts[2] in ['day', 'days']:
                    return (today + timedelta(days=num)).strftime('%d/%m/%Y')
                elif parts[2] in ['week', 'weeks']:
                    return (today + timedelta(weeks=num)).strftime('%d/%m/%Y')
        elif date_str.startswith('on '):
            # Handle day names like "on monday"
            day_name = date_str[3:].capitalize()
            try:
                weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                          'Friday', 'Saturday', 'Sunday'].index(day_name)
                days_ahead = (weekday - today.weekday()) % 7
                if days_ahead == 0:  # Today is that day
                    days_ahead = 7  # Make it next week
                return (today + timedelta(days=days_ahead)).strftime('%d/%m/%Y')
            except ValueError:
                pass
        return None

    def parse_time(self, time_str: str) -> Optional[str]:
        """Parse various time formats into HH:MM"""
        time_str = time_str.lower().strip()
        
        # Handle "5pm", "3:30am" etc.
        if 'am' in time_str or 'pm' in time_str:
            try:
                is_pm = 'pm' in time_str
                time_str = time_str.replace('am', '').replace('pm', '').strip()
                
                if ':' in time_str:
                    hours, minutes = time_str.split(':')
                else:
                    hours, minutes = time_str, '00'
                
                hours = int(hours)
                minutes = int(minutes)
                
                # Convert to 24-hour format
                if is_pm and hours < 12:
                    hours += 12
                elif not is_pm and hours == 12:
                    hours = 0
                    
                return f"{hours:02d}:{minutes:02d}"
            except:
                return None
        
        # Handle "half past 5", "quarter to 6" etc.
        time_expr = re.search(r'(half|quarter|\d+)\s*(past|to)\s*(\d+)', time_str)
        if time_expr:
            amount, direction, hour = time_expr.groups()
            hour = int(hour)
            
            if direction == 'to':
                hour -= 1
                minutes = 60 - (15 if amount == 'quarter' else 30 if amount == 'half' else int(amount))
            else:
                minutes = 15 if amount == 'quarter' else 30 if amount == 'half' else int(amount)
            
            if hour < 0:
                hour = 23
            return f"{hour:02d}:{minutes:02d}"
        
        # Handle simple formats like "5", "5:30"
        if re.match(r'^\d{1,2}(:\d{2})?$', time_str):
            if ':' in time_str:
                hours, minutes = time_str.split(':')
            else:
                hours, minutes = time_str, '00'
            
            hours = int(hours)
            minutes = int(minutes)
            
            if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
                return None
                
            return f"{hours:02d}:{minutes:02d}"
        
        return None

    def extract_booking_info(self, text: str) -> Tuple[Dict, List]:
        """Improved information extraction with better station handling"""
        print(f"\nExtracting info from: '{text}'")
        
        extracted = {}
        missing = []
        text_lower = text.lower()
        
        # 1. Try explicit "from X to Y" pattern
        from_to_match = re.search(
            r'(?:from|departing)\s+(.+?)\s+(?:to|going to|arriving in|arriving at)\s+(.+?)(?:\s+(?:on|at)\s+|$)', 
            text_lower, 
            re.IGNORECASE
        )
        
        if from_to_match:
            from_text = from_to_match.group(1).strip()
            to_text = from_to_match.group(2).strip()
            
            print(f"From-to pattern matched - From: '{from_text}', To: '{to_text}'")
            
            from_station = self.match_station(from_text)
            to_station = self.match_station(to_text.split()[0])  # Take first word only
            
            if from_station:
                extracted['origin'] = self.station_codes.get(from_station.lower())
            if to_station:
                extracted['destination'] = self.station_codes.get(to_station.lower())
        
        # 2. Extract date and time
        date_patterns = [
            r'(today|tomorrow|day after tomorrow)',
            r'(next|in \d+) (week|weeks|day|days)',
            r'on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?'
        ]
        
        date_match = re.search('|'.join(date_patterns), text_lower)
        if date_match:
            date_str = date_match.group()
            extracted['date'] = self.parse_relative_date(date_str) or date_str
        
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}(?:am|pm)',
            r'\d{1,2}(?::\d{2})?(?:am|pm)',
            r'(half|quarter|\d+) (past|to) \d+'
        ]
        
        time_match = re.search('|'.join(time_patterns), text_lower)
        if time_match:
            time_str = time_match.group()
            extracted['time'] = self.parse_time(time_str)
        
        # 3. Check passenger numbers
        adults_match = re.search(r'(\d+)\s*(adult|people|person|passenger)', text)
        if adults_match:
            extracted['adults'] = adults_match.group(1)
        
        children_match = re.search(r'(\d+)\s*(child|children|kid|kids)', text)
        if children_match:
            extracted['children'] = children_match.group(1)
        
        # 4. Check for return journey
        if any(word in text.lower() for word in ['return', 'round trip', 'coming back']):
            extracted['is_return'] = True
        
        # 5. Check required fields
        required_fields = ['origin', 'destination', 'date', 'time']
        for field in required_fields:
            if field not in extracted:
                missing.append(field)
                
        # Set defaults
        extracted.setdefault('adults', '1')
        extracted.setdefault('children', '0')
        
        print(f"Extracted: {extracted}")
        print(f"Missing: {missing}")
        
        return extracted, missing

    def generate_missing_prompt(self, missing_fields: List[str]) -> str:
        """Generate a prompt asking for missing information"""
        field_names = {
            'origin': "departure station",
            'destination': "destination station",
            'date': "travel date",
            'time': "travel time",
            'return_date': "return date",
            'return_time': "return time"
        }
        
        if not missing_fields:
            return ""
        
        if len(missing_fields) == 1:
            return f"Could you please provide the {field_names.get(missing_fields[0], missing_fields[0])}?"
        else:
            fields = [field_names.get(f, f) for f in missing_fields]
            if len(fields) == 2:
                return f"Could you please provide the {fields[0]} and {fields[1]}?"
            else:
                return f"Could you please provide the {', '.join(fields[:-1])}, and {fields[-1]}?"

    def validate_date(self, date_str: str) -> bool:
        """Validate date format DD/MM/YYYY or relative date expressions"""
        # First try relative date parsing
        relative_date = self.parse_relative_date(date_str)
        if relative_date:
            return True
        
        # Then try standard date format
        try:
            day, month, year = date_str.split('/')
            if len(day) == 2 and len(month) == 2 and len(year) == 4:
                datetime.strptime(date_str, '%d/%m/%Y')
                return True
        except ValueError:
            return False
        return False

    def validate_time(self, time_str: str) -> bool:
        """Validate time format HH:MM"""
        try:
            datetime.strptime(time_str, '%H:%M')
            return True
        except ValueError:
            return False

    def detect_intent(self, text: str) -> str:
        """Enhanced intent detection with more booking-related phrases"""
        text = text.lower()
        
        # Greeting detection
        if any(word in text for word in ['hello', 'hi', 'hey']):
            return "greeting"
        
        # Farewell detection
        if any(word in text for word in ['bye', 'goodbye', 'exit', 'quit']):
            return "farewell"
        
        # Booking detection - expanded list of triggers
        booking_triggers = [
            'book', 'ticket', 'journey', 'travel', 'train',
            'schedule', 'timetable', 'from', 'to', 'depart', 'arrive',
            'i want to go', 'i need to go', 'going to', 'travel to',
            'i would like to book', 'i want to book', 'need a ticket'
        ]
        if any(trigger in text for trigger in booking_triggers):
            return "booking"
        
        # Delay detection
        delay_triggers = ['delay', 'late', 'arrival', 'on time', 'status', 'how long', 'when will']
        if any(trigger in text for trigger in delay_triggers):
            return "delay_inquiry"
        
        return "unknown"

    def generate_response(self, user_input: str) -> str:
        """Generate appropriate response based on current state"""
        self.log_conversation(user_input, "")
        
        # Detect intent
        intent = self.detect_intent(user_input)
        
        # Handle farewell immediately
        if intent == "farewell":
            self.current_state = self.STATES['FAREWELL']
            return random.choice(self.farewells)
        
        # State machine logic
        if self.current_state == self.STATES['GREETING']:
            # Skip greeting if user already made a request
            if intent == "booking":
                self.current_state = self.STATES['PARSE_BOOKING']
                # Extract info right away
                extracted_info, missing_fields = self.extract_booking_info(user_input)
                for key, value in extracted_info.items():
                    if value:
                        self.booking_details[key] = value
                
                if missing_fields:
                    self.missing_fields = missing_fields
                    self.current_state = self.STATES['GET_MISSING_INFO']
                    prompt = self.generate_missing_prompt(missing_fields)
                    return f"I need a bit more information to help you. {prompt}"
                else:
                    self.current_state = self.STATES['CONFIRM_BOOKING']
                    return self.process_booking()
                    
            elif intent == "delay_inquiry":
                self.current_state = self.STATES['DELAY_INQUIRY']
                return "I can help with delay predictions. Which train are you on? (Norwich to London or London to Norwich)"
            else:
                self.current_state = self.STATES['GET_OPTION']
                return f"{random.choice(self.greetings)}\n{self.options_prompt}"
        
        # Rest of the state machine remains the same...
        elif self.current_state == self.STATES['GET_OPTION']:
            # ... (keep existing code)
            if intent == "delay_inquiry":
                self.current_state = self.STATES['DELAY_INQUIRY']
                return "I can help with delay predictions. Which train are you on? (Norwich to London or London to Norwich)"
            elif intent == "booking" or "1" in user_input:
                self.current_state = self.STATES['PARSE_BOOKING']
                return "Please tell me about your journey. For example: 'I want to travel from Norwich to London tomorrow at 3pm'"
            elif "2" in user_input:
                self.current_state = self.STATES['DELAY_INQUIRY']
                return "I can help with delay predictions. Which train are you on? (Norwich to London or London to Norwich)"
            elif "3" in user_input or intent == "farewell":
                self.current_state = self.STATES['FAREWELL']
                return random.choice(self.farewells)
            else:
                return "I didn't understand that. Please choose:\n1. Book tickets\n2. Check for delays\n3. Exit"
        
        elif self.current_state == self.STATES['PARSE_BOOKING']:
            # Extract information from user input
            extracted_info, missing_fields = self.extract_booking_info(user_input)
            
            # Update booking details with extracted info
            for key, value in extracted_info.items():
                if value:  # Only update if we got a value
                    self.booking_details[key] = value
            
            if not missing_fields:
                # All required info is present
                self.current_state = self.STATES['CONFIRM_BOOKING']
                return self.process_booking()
            else:
                # Ask for missing information
                self.missing_fields = missing_fields
                self.current_state = self.STATES['GET_MISSING_INFO']
                prompt = self.generate_missing_prompt(missing_fields)
                return f"I need a bit more information to help you. {prompt}"
        
        elif self.current_state == self.STATES['GET_MISSING_INFO']:
            # Try to extract the missing information from the response
            extracted_info, _ = self.extract_booking_info(user_input)
            
            # Update booking details with any new info
            for key, value in extracted_info.items():
                if value and key in self.missing_fields:
                    self.booking_details[key] = value
                    self.missing_fields.remove(key)
            
            if not self.missing_fields:
                # Got all missing info
                self.current_state = self.STATES['CONFIRM_BOOKING']
                return self.process_booking()
            else:
                # Still missing some info
                prompt = self.generate_missing_prompt(self.missing_fields)
                return f"I still need some more information. {prompt}"
        
        elif self.current_state == self.STATES['CONFIRM_BOOKING']:
            if user_input.lower() in ['yes', 'y']:
                result = self.process_booking(confirm=True)
                self.current_state = self.STATES['GET_OPTION']  # Return to main menu
                return result + "\n\n" + self.options_prompt
            elif user_input.lower() in ['no', 'n']:
                self.reset_booking()
                self.current_state = self.STATES['GET_OPTION']
                return "Okay, let's start over. " + self.options_prompt
            else:
                return "Please answer with 'yes' or 'no'"
        
        elif self.current_state == self.STATES['DELAY_INQUIRY']:
            if "norwich" in user_input.lower() and "london" in user_input.lower():
                self.booking_details['route'] = "Norwich-LST"
                self.current_state = self.STATES['GET_CURRENT_STATION']
                return "At which station is the train currently? (Norwich, Diss, Ipswich, or London Liverpool Street)"
            elif "london" in user_input.lower() and "norwich" in user_input.lower():
                self.booking_details['route'] = "LST-Norwich"
                self.current_state = self.STATES['GET_CURRENT_STATION']
                return "At which station is the train currently? (London Liverpool Street, Ipswich, Diss, or Norwich)"
            else:
                return "Please specify 'Norwich to London' or 'London to Norwich'"
        
        elif self.current_state == self.STATES['GET_CURRENT_STATION']:
            if self.booking_details.get('route'):
                stations = [self.get_station_name(code) for code in self.station_sequences.get(self.booking_details['route'], [])]
                station = self.match_station(user_input)
                if station:
                    station_code = self.get_station_code(station)
                    if station_code and station_code in self.station_sequences.get(self.booking_details['route'], []):
                        self.booking_details['current_station'] = station_code
                        self.current_state = self.STATES['GET_DELAY_DURATION']
                        return "How many minutes has the train been delayed?"
                return f"Station not recognized. Please try again with one of: {', '.join(stations)}"
            return "Route not recognized. Please specify 'Norwich to London' or 'London to Norwich'"
        
        elif self.current_state == self.STATES['GET_DELAY_DURATION']:
            match = re.search(r'(\d+)', user_input)
            if match:
                delay = int(match.group(1))
                prediction = self.predict_arrival(
                    self.booking_details['route'],
                    self.booking_details['current_station'],
                    delay
                )
                if prediction:
                    self.current_state = self.STATES['GET_OPTION']  # Return to main menu
                    station_name = self.get_station_name('LST')
                    return (
                        f"With {delay} minute delay at {self.get_station_name(self.booking_details['current_station'])}, "
                        f"your train should arrive at {station_name} in about {prediction} minutes.\n\n"
                        + self.options_prompt
                    )
                return "Couldn't calculate arrival time. Please check your information."
            return "Please specify the delay in minutes (e.g., '15 minutes')."

    def reset_booking(self):
        """Reset booking details"""
        self.booking_details = {
            'origin': None,
            'destination': None,
            'date': None,
            'time': None,
            'adults': '1',
            'children': '0',
            'is_return': False,
            'return_date': None,
            'return_time': None
        }
        self.missing_fields = []

    def process_booking(self, confirm=False):
        """Process the booking and return results"""
        # Build summary message
        summary = "Here's your journey summary:\n"
        summary += f"From: {self.get_station_name(self.booking_details['origin'])}\n"
        summary += f"To: {self.get_station_name(self.booking_details['destination'])}\n"
        summary += f"Date: {self.booking_details['date']}\n"
        summary += f"Time: {self.booking_details['time']}\n"
        summary += f"Adults: {self.booking_details['adults']}\n"
        summary += f"Children: {self.booking_details['children']}\n"
        
        if self.booking_details['is_return']:
            if self.booking_details.get('return_date') and self.booking_details.get('return_time'):
                summary += f"Return Date: {self.booking_details['return_date']}\n"
                summary += f"Return Time: {self.booking_details['return_time']}\n"
            else:
                self.missing_fields.extend(['return_date', 'return_time'])
                self.current_state = self.STATES['GET_MISSING_INFO']
                return "You mentioned this is a return journey. " + self.generate_missing_prompt(['return_date', 'return_time'])
        
        if not confirm:
            summary += "\nWould you like me to find the best tickets for this journey? (yes/no)"
            return summary
        
        # If confirmed, proceed with web scraping
        summary += "\nSearching for the best tickets...\n"
        
        try:
            # Build URL and launch browser
            url, target_hour, target_minute = self.build_url()
            
            driver = webdriver.Chrome()
            driver.get(url)
            
            self.accept_cookies_if_present(driver)
            
            if self.booking_details['is_return']:
                # Handle return journey
                outward_time, outward_price = self.select_cheapest_outward_journey(driver, target_hour, target_minute)
                return_time, return_price = self.select_return_journey(driver, self.booking_details['return_time'])
                
                if outward_time and return_time:
                    total_price = outward_price + return_price
                    summary += f"\nTickets found!\n"
                    summary += f"Outward: {outward_time} (£{outward_price:.2f})\n"
                    summary += f"Return: {return_time} (£{return_price:.2f})\n"
                    summary += f"Total Price: £{total_price:.2f}\n"
                    summary += f"\nYou can book here: {url}"
                else:
                    summary += "Sorry, I couldn't find suitable tickets for your journey."
            else:
                # Handle one-way journey
                best_match = self.extract_cheapest_ticket(driver, target_hour, target_minute)
                driver.quit()
                
                if best_match:
                    summary += f"\nCheapest ticket found:\n"
                    summary += f"Time: {best_match['dep_time']}\n"
                    summary += f"Price: £{best_match['price']:.2f}\n"
                    summary += f"\nYou can book here: {url}"
                else:
                    summary += "Sorry, I couldn't find suitable tickets for your journey."
            
            self.current_state = self.STATES['FAREWELL']
            return summary
            
        except Exception as e:
            return f"An error occurred while searching for tickets: {str(e)}"

    def predict_arrival(self, route: str, current_station: str, delay: int) -> Optional[int]:
        """Predict arrival time at destination with delay"""
        try:
            # Get station sequence for this route
            if route not in self.station_sequences:
                return None
                
            stations = self.station_sequences[route]
            
            # Find current position in route
            try:
                current_idx = stations.index(current_station)
            except ValueError:
                return None  # Station not on this route
                
            # Find destination (London Liverpool Street)
            try:
                dest_idx = stations.index('LST')
            except ValueError:
                return None  # No LST in route
                
            # Calculate schedule times based on historical data
            schedule_times = self.calculate_schedule_times(route, stations)
            
            if not schedule_times:
                return None
                
            remaining_time = schedule_times[dest_idx] - schedule_times[current_idx]
            
            # Adjust for expected additional delays based on historical data
            additional_delay = 0
            for i in range(current_idx, dest_idx):
                from_stn = stations[i]
                to_stn = stations[i+1]
                pair = (from_stn, to_stn)
                
                if pair in self.delay_stats:
                    # Use median delay between these stations
                    additional_delay += self.delay_stats[pair]['median_delay']
            
            total_delay = delay + additional_delay
            
            # Return total estimated time remaining
            return int(remaining_time + total_delay)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def calculate_schedule_times(self, route, stations):
        """Calculate typical schedule times between stations"""
        if route not in self.delay_data or not isinstance(self.delay_data[route], pd.DataFrame):
            return None
            
        df = self.delay_data[route]
        schedule_times = [0]  # Start at 0 minutes for first station
        
        for i in range(1, len(stations)):
            from_stn = stations[i-1]
            to_stn = stations[i]
            
            # Get typical travel time between these stations
            mask = (df['location'] == from_stn)
            departures = df[mask]['actual_departure'].dropna()
            
            mask = (df['location'] == to_stn)
            arrivals = df[mask]['actual_arrival'].dropna()
            
            if not departures.empty and not arrivals.empty:
                # Convert times to minutes since midnight for calculation
                departures_min = departures.apply(lambda x: x.hour * 60 + x.minute)
                arrivals_min = arrivals.apply(lambda x: x.hour * 60 + x.minute)
                
                # Calculate median travel time in minutes
                typical_time = (arrivals_min.median() - departures_min.median())
                
                # Handle overnight trains (negative time difference)
                if typical_time < 0:
                    typical_time += 1440  # Add 24 hours in minutes
                    
                schedule_times.append(schedule_times[-1] + typical_time)
            else:
                # Fallback: use 30 minutes between stations if no data
                schedule_times.append(schedule_times[-1] + 30)
        
        return schedule_times

    def round_down_to_quarter(self, minute):
        return str((int(minute) // 15) * 15).zfill(2)

    def build_url(self):
        """Build the booking URL from collected details"""
        origin = self.booking_details['origin']
        destination = self.booking_details['destination']
        date_str = self.booking_details['date']
        time_str = self.booking_details['time']
        adults = self.booking_details['adults']
        children = self.booking_details['children']
        return_date_str = self.booking_details['return_date']
        return_time_str = self.booking_details['return_time']

        day, month, year = date_str.split("/")
        hour, minute = time_str.split(":")
        minute = self.round_down_to_quarter(minute)
        leaving_date = f"{day}{month}{year[2:]}"
        leaving_hour = hour.zfill(2)

        params = {
            "origin": origin,
            "destination": destination,
            "leavingType": "departing",
            "leavingDate": leaving_date,
            "leavingHour": leaving_hour,
            "leavingMin": minute,
            "adults": adults,
            "children": children,
            "extraTime": "0"
        }

        if return_date_str and return_time_str:
            r_day, r_month, r_year = return_date_str.split("/")
            r_hour, r_minute = return_time_str.split(":")
            r_minute = self.round_down_to_quarter(r_minute)
            return_date = f"{r_day}{r_month}{r_year[2:]}"
            return_hour = r_hour.zfill(2)

            params.update({
                "type": "return",
                "returnType": "departing",
                "returnDate": return_date,
                "returnHour": return_hour,
                "returnMin": r_minute
            })
        else:
            params["type"] = "single"

        url = "https://www.nationalrail.co.uk/journey-planner/?" + urlencode(params)
        return url, int(hour), int(minute)

    def accept_cookies_if_present(self, driver):
        try:
            wait = WebDriverWait(driver, 10)
            reject_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Reject All')]")))
            reject_btn.click()
            print(" Cookie popup rejected.")
        except:
            print(" No cookie popup appeared or already handled.")

    def extract_cheapest_ticket(self, driver, target_hour, target_minute):
        time.sleep(10)
        cards = driver.find_elements(By.CSS_SELECTOR, "section[data-testid^='result-card-section-outward']")
        cheapest_price = float("inf")
        best_match = None

        for card in cards:
            try:
                dep_time_elem = card.find_element(By.CSS_SELECTOR, "time[datetime]")
                dep_time_str = dep_time_elem.text.strip()
                dep_time = datetime.strptime(dep_time_str, "%H:%M")
                dep_minutes = dep_time.hour * 60 + dep_time.minute
                target_minutes = target_hour * 60 + target_minute
                diff = abs(dep_minutes - target_minutes)

                if diff > 30:
                    continue

                price_elem = card.find_element(By.XPATH, ".//span[contains(text(), '£')]")
                price_text = price_elem.text.strip().replace("£", "")
                price = float(price_text)

                if price < cheapest_price:
                    cheapest_price = price
                    best_match = {
                        "dep_time": dep_time_str,
                        "price": price
                    }
            except Exception:
                continue

        return best_match

    def select_cheapest_outward_journey(self, driver, target_hour, target_minute):
        """Select cheapest outward journey within time window"""
        time.sleep(10)
        cards = driver.find_elements(By.CSS_SELECTOR, "section[data-testid^='result-card-section-outward']")
        best_card = None
        lowest_price = float('inf')
        best_time = None

        for card in cards:
            try:
                dep_time_elem = card.find_element(By.CSS_SELECTOR, "time[datetime]")
                dep_time_str = dep_time_elem.text.strip()
                dep_time = datetime.strptime(dep_time_str, "%H:%M")
                dep_minutes = dep_time.hour * 60 + dep_time.minute
                target_minutes = target_hour * 60 + target_minute
                diff = abs(dep_minutes - target_minutes)

                if diff > 60:  # 60-minute window
                    continue

                price_elem = card.find_element(By.XPATH, ".//span[contains(text(), '£')]")
                price_text = price_elem.text.strip().replace("£", "")
                price = float(price_text)

                if price < lowest_price:
                    lowest_price = price
                    best_card = card
                    best_time = dep_time_str

            except Exception:
                continue

        if best_card:
            try:
                button = best_card.find_element(By.XPATH, ".//input[@type='button']")
                driver.execute_script("arguments[0].click();", button)
                return best_time, lowest_price
            except Exception:
                return None, None
        return None, None

    def select_return_journey(self, driver, return_time_str):
        """Select return journey within time window"""
        user_return_time = datetime.strptime(return_time_str, "%H:%M")
        time_window = timedelta(minutes=60)
        time.sleep(10)
        journey_sections = driver.find_elements(By.CSS_SELECTOR, 'section[id^="inward-"]')

        best_option = None
        best_price = float('inf')
        closest_time_diff = timedelta.max
        best_time = None

        for section in journey_sections:
            try:
                time_elem = section.find_element(By.CSS_SELECTOR, 'time[class*="bAcVR"]')
                return_time = datetime.strptime(time_elem.text.strip(), "%H:%M")
                time_diff = abs(return_time - user_return_time)

                if time_diff <= time_window:
                    price_elem = section.find_element(By.XPATH, ".//div[contains(@id,'price')]/div/span[2]")
                    price = float(price_elem.text.replace("£", "").strip())

                    if price < best_price or (price == best_price and time_diff < closest_time_diff):
                        best_price = price
                        best_time = time_elem.text.strip()
                        closest_time_diff = time_diff
                        best_option = section

            except Exception:
                continue

        if best_option:
            try:
                select_btn = best_option.find_element(By.XPATH, './/input[@type="button"]')
                driver.execute_script("arguments[0].click();", select_btn)
                return best_time, best_price
            except Exception:
                return None, None
        return None, None


if __name__ == "__main__":
    print("Initializing Train Ticket Chatbot...")
    chatbot = TrainTicketChatbotWithBooking()

    print("\nLoading and analyzing delay data...")
    chatbot.load_delay_data()

    print("\nWelcome to TrainBot! Type 'exit' to quit at any time.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("TrainBot:", random.choice(chatbot.farewells))
            break
        
        response = chatbot.generate_response(user_input)
        print("TrainBot:", response)
import re
import collections
import collections.abc
from ChatBotTrainMixed import compare_ticket_prices

collections.Mapping = collections.abc.Mapping
collections.Sequence = collections.abc.Sequence
collections.Iterable = collections.abc.Iterable
from experta import KnowledgeEngine, Rule, Fact, MATCH
from fuzzywuzzy import fuzz
import sqlite3
import random
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
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
from dateutil.parser import parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Initialize spaCy for NLP
try:
    nlp = spacy.load('en_core_web_sm')
    print("NLP model loaded successfully.")
except:
    print("spaCy not available. Some features will be limited.")
    nlp = None


class RandomForestDelayPredictor:
    """Random Forest delay predictor with dynamic station handling"""

    def __init__(self, station_sequences):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.station_sequences = station_sequences
        self.station_to_idx = self._create_station_mapping()
        self.is_trained = False
        self.feature_names = ['day_of_week', 'hour_of_day', 'station_idx', 'route_type']

    def _create_station_mapping(self):
        """Create dynamic station index mapping"""
        station_map = {}
        for route, stations in self.station_sequences.items():
            for idx, station in enumerate(stations):
                station_map[(route, station)] = idx
        return station_map

    def preprocess_data(self, df):
        """Prepare data with dynamic station handling"""
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Convert times to datetime if they aren't already
        if 'planned_departure' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['planned_departure']):
            try:
                df['planned_departure'] = pd.to_datetime(df['planned_departure'], errors='coerce')
            except:
                pass

        # Calculate delay if not already present
        if 'delay_minutes' not in df.columns and 'planned_departure' in df.columns and 'actual_departure' in df.columns:
            df['delay_minutes'] = (
                    (df['actual_departure'] - df['planned_departure']).dt.total_seconds() / 60
            )

        # Add station position
        df['station_idx'] = df.apply(
            lambda x: self.station_to_idx.get((x['route'], x['location']), -1),
            axis=1
        )

        # Feature engineering
        df['day_of_week'] = df['planned_departure'].dt.dayofweek
        df['hour_of_day'] = df['planned_departure'].dt.hour
        df['route_type'] = df['route'].map({
            'Norwich-LST': 0,
            'LST-Norwich': 1
        })

        # Drop rows with missing critical data
        df = df.dropna(subset=['delay_minutes', 'station_idx'])

        return df

    def train(self, df):
        processed = self.preprocess_data(df)
        X = processed[self.feature_names]
        y = processed['delay_minutes']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        print(f"Model trained on {len(X_train)} samples. Validation MAE: {mae:.2f} minutes")

    def predict(self, route, current_station):
        """Make prediction using current conditions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        station_idx = self.station_to_idx.get((route, current_station), -1)
        if station_idx == -1:
            return None

        # Create input features matching the training structure exactly
        input_features = pd.DataFrame([{
            'day_of_week': datetime.now().weekday(),
            'hour_of_day': datetime.now().hour,
            'station_idx': station_idx,
            'route_type': 0 if route == 'Norwich-LST' else 1
        }])[self.feature_names]  # Ensure same column order as training

        return float(self.model.predict(input_features)[0])


class DelayReasoningEngine(KnowledgeEngine):
    """Rule-based system for delay predictions"""

    def __init__(self, station_sequences, delay_stats):
        super().__init__()
        self.station_sequences = station_sequences
        self.delay_stats = delay_stats

    @Rule(Fact(route=MATCH.r),
          Fact(current_station=MATCH.s),
          Fact(reported_delay=MATCH.d))
    def predict_delay(self, r, s, d):
        # Calculate cumulative delay using rules
        stations = self.station_sequences[r]
        current_idx = stations.index(s)
        total_delay = float(d)

        # Add median delays between current station and destination
        for i in range(current_idx, len(stations) - 1):
            from_stn = stations[i]
            to_stn = stations[i + 1]
            pair = (from_stn, to_stn)
            if pair in self.delay_stats:
                total_delay += self.delay_stats[pair]['median_delay']

        self.declare(Fact(predicted_delay=total_delay))


class TrainTicketChatbotWithBooking:
    def __init__(self):
        """Initialize the chatbot with all required components"""
        # Initialize NLP processors
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Now safely initialize NLP components
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

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
            'GET_DELAY_DURATION': 8,
            'GET_DESTINATION': 9,
            'ASK_RETURN': 10
        }

        self.current_state = self.STATES['GREETING']
        self.booking_details = {
            'origin': None,
            'destination': None,
            'date': None,
            'time': None,
            'adults': '1',
            'children': '0',
            # 'is_return': False,
            # 'return_date': None,
            # 'return_time': None,
            'confirmed': False
        }
        self.missing_fields = []

        self.stations = []
        self.station_codes = {}

        # Load station data from CSV
        self.load_station_data('stations.csv')  # Update with your CSV path

        # Set up NLP
        if nlp:
            self.matcher = Matcher(nlp.vocab)
            self.initialize_nlp_matchers()

        # Initialize database
        self.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.create_tables()

        # Load and process delay data
        self.load_delay_data()

        self.delay_engine = DelayReasoningEngine(
            station_sequences=self.station_sequences,
            delay_stats=self.delay_stats
        )
        # Initialize predictor with dynamic stations
        self.delay_predictor = RandomForestDelayPredictor(
            station_sequences=self.station_sequences
        )
        # Train model

        self._train_delay_model()
        self.delay_details = {}
        self.normal_travel_times = {}
        for route in self.station_sequences:
            self.normal_travel_times[route] = self.calculate_normal_travel_times(route)

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
        """Properly extract station sequences from CSV files"""
        self.station_sequences = {'Norwich-LST': [], 'LST-Norwich': []}

        # Process Norwich to London file
        if isinstance(self.delay_data['Norwich-LST'], pd.DataFrame):
            df = self.delay_data['Norwich-LST']
            # Group by train ID and get unique station order
            sequences = df.groupby('rid').apply(
                lambda x: x.sort_values('planned_departure_time')['location'].unique().tolist()
            )
            if len(sequences) > 0:
                self.station_sequences['Norwich-LST'] = sequences.iloc[0]  # Take first sequence

        # Process London to Norwich file
        if isinstance(self.delay_data['LST-Norwich'], pd.DataFrame):
            df = self.delay_data['LST-Norwich']
            # Group by train ID and get unique station order
            sequences = df.groupby('rid').apply(
                lambda x: x.sort_values('gbtt_ptd')['location'].unique().tolist()
            )
            if len(sequences) > 0:
                self.station_sequences['LST-Norwich'] = sequences.iloc[0]  # Take first sequence

        print(f"Final station sequences: {self.station_sequences}")

    def preprocess_text(self, text: str) -> List[str]:
        """Enhanced text preprocessing with stemming and lemmatization"""
        try:
            # Tokenize and lowercase
            tokens = nltk.word_tokenize(text.lower())

            # Remove stopwords and punctuation
            tokens = [token for token in tokens
                      if token not in self.stop_words and token.isalpha()]

            processed_tokens = []
            for token in tokens:
                try:
                    # Lemmatize first (more accurate)
                    lemma = self.lemmatizer.lemmatize(token)
                    # Then stem (more aggressive)
                    stem = self.stemmer.stem(lemma)

                    # Use stem if it's significantly shorter, else use lemma
                    if len(stem) < len(lemma) - 1:
                        processed_tokens.append(stem)
                    else:
                        processed_tokens.append(lemma)
                except:
                    processed_tokens.append(token)

            return processed_tokens
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return []

    def parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats into DD/MM/YYYY"""
        try:
            # Try different date formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d',
                        '%B %d %Y', '%b %d %Y', '%d %B %Y', '%d %b %Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%d/%m/%Y')
                except ValueError:
                    continue

            # Handle relative dates
            today = datetime.now().date()
            if date_str.lower() == 'today':
                return today.strftime('%d/%m/%Y')
            if date_str.lower() == 'tomorrow':
                return (today + timedelta(days=1)).strftime('%d/%m/%Y')

            # Handle "next <dayname>"
            if date_str.lower().startswith('next '):
                day_name = date_str[5:].capitalize()
                weekdays = ['Monday', 'Tuesday', 'Wednesday',
                            'Thursday', 'Friday', 'Saturday', 'Sunday']
                if day_name in weekdays:
                    days_ahead = (weekdays.index(day_name) - today.weekday()) % 7
                    if days_ahead <= 0:  # If today is that day, go to next week
                        days_ahead += 7
                    return (today + timedelta(days=days_ahead)).strftime('%d/%m/%Y')

            return None
        except Exception:
            return None

    def _train_delay_model(self):
        """Train model with all available data"""
        try:
            # Combine data from both routes
            combined_data = pd.concat([
                self.delay_data['Norwich-LST'],
                self.delay_data['LST-Norwich']
            ], ignore_index=True)

            print("\nColumns available for training:", combined_data.columns.tolist())

            # Check if we have enough data
            if len(combined_data) < 100:
                raise ValueError(f"Not enough training data (only {len(combined_data)} samples)")

            # Train the model
            self.delay_predictor.train(combined_data)
            print("Random Forest model trained successfully")

        except Exception as e:
            print(f"\nError training Random Forest model: {str(e)}")
            print("Will use rule-based predictions only")
            self.delay_predictor.is_trained = False

    def preprocess_delay_data(self):
        """Clean and prepare delay data for analysis"""
        for route in self.delay_data:
            if isinstance(self.delay_data[route], pd.DataFrame):
                df = self.delay_data[route].copy()
                time_cols = ['planned_arrival', 'planned_departure', 'actual_arrival', 'actual_departure']
                for col in time_cols:
                    if col in df:
                        df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce').dt.time
            # Standardize column names across both routes
            if route == 'Norwich-LST':
                # Rename columns to standard names
                column_map = {
                    'planned_arrival_time': 'planned_arrival',
                    'planned_departure_time': 'planned_departure',
                    'actual_arrival_time': 'actual_arrival',
                    'actual_departure_time': 'actual_departure'
                }
                df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

                # Remove duplicate column if exists
                if 'late_canc_reason.1' in df.columns:
                    df = df.drop(columns=['late_canc_reason.1'])

            elif route == 'LST-Norwich':
                # Rename columns to standard names
                column_map = {
                    'gbtt_pta': 'planned_arrival',
                    'gbtt_ptd': 'planned_departure',
                    'actual_ta': 'actual_arrival',
                    'actual_td': 'actual_departure'
                }
                df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

            # Ensure we have the required columns
            required_cols = ['planned_departure', 'actual_departure', 'location']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {route} data: {missing_cols}")
                continue

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
                        try:
                            df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time
                        except:
                            df[col] = pd.NA

            # Calculate delay minutes (departure delay)
            df['delay_minutes'] = 0
            mask = df['actual_departure'].notna() & df['planned_departure'].notna()

            # Calculate delay in minutes
            df.loc[mask, 'delay_minutes'] = df[mask].apply(
                lambda x: (datetime.combine(datetime.today(), x['actual_departure']) -
                           datetime.combine(datetime.today(), x['planned_departure'])).total_seconds() / 60,
                axis=1
            )

            # Filter out extreme delays (more than 120 minutes)
            df = df[df['delay_minutes'].abs() <= 120]

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

        for i in range(len(stations) - 1):
            from_stn = stations[i]
            to_stn = stations[i + 1]
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
        self.stations = []  # List of station names
        self.station_codes = {}  # Dictionary mapping names to codes

        try:
            # Try reading CSV with different possible delimiters
            try:
                df = pd.read_csv(csv_path)
            except:
                # Try with different encodings if first attempt fails
                try:
                    df = pd.read_csv(csv_path, encoding='latin1')
                except:
                    df = pd.read_csv(csv_path, encoding='utf-16')

            print(f"Successfully loaded CSV with {len(df)} rows")

            # Process each row
            for _, row in df.iterrows():
                try:
                    # Get station name - try different possible column names
                    station_name = None
                    for col in ['name', 'station_name', 'location', 'Station_Name']:
                        if col in row:
                            station_name = str(row[col]).strip()
                            if station_name and station_name != '\\N':
                                break

                    if not station_name:
                        continue

                    # Get station code - try different possible column names
                    tiploc = None
                    for col in ['alpha3', 'tiploc', 'station_code', 'code']:
                        if col in row:
                            tiploc = str(row[col]).strip()
                            if tiploc and tiploc != '\\N':
                                break

                    if not tiploc:
                        continue

                    # Add to stations list if not already present
                    if station_name not in self.stations:
                        self.stations.append(station_name)

                    # Add primary mapping
                    self.station_codes[station_name.lower()] = tiploc

                    # Handle aliases if available
                    if 'alias' in row:
                        alias = str(row['alias']).strip()
                        if alias and alias != '\\N':
                            self.station_codes[alias.lower()] = tiploc

                    # Handle alternative name columns
                    if 'alternative_name' in row:
                        alt_name = str(row['alternative_name']).strip()
                        if alt_name and alt_name != '\\N':
                            self.station_codes[alt_name.lower()] = tiploc

                except Exception as e:
                    print(f"Error processing row {_}: {e}")
                    continue

            print(f"Loaded {len(self.stations)} stations total")
            print(f"Sample stations: {self.stations[:5]}")

        except Exception as e:
            print(f"Error loading station CSV: {e}")

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

    def initialize_nlp_matchers(self):
        if not nlp:
            print("spaCy not available, skipping NLP matcher initialization.")
            return
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("DEPARTURE", [[{"LOWER": {"IN": ["from", "departing"]}}, {"ENT_TYPE": "GPE"}]])
        self.matcher.add("DESTINATION", [[{"LOWER": {"IN": ["to", "going"]}}, {"ENT_TYPE": "GPE"}]])

    def match_station(self, text: str) -> Optional[str]:
        """Improved matching with common abbreviations"""
        text = text.lower().strip()

        # Handle common abbreviations dynamically
        abbrev_map = {
            'london': ['liverpool street', 'lst'],
            'norwich': ['nrwi', 'nrwh']
        }

        # Check full names first
        for name in self.station_codes.keys():
            if text == name.lower():
                return name

            # Check against common abbreviations
            for station, abbrevs in abbrev_map.items():
                if station in name.lower() and any(abbr in text for abbr in abbrevs):
                    return name

        # Then try fuzzy matching
        best_match = None
        best_score = 0
        for name in self.station_codes.keys():
            score = fuzz.partial_ratio(text, name.lower())
            if score > best_score and score > 80:
                best_score = score
                best_match = name

        return best_match

    def get_station_code(self, station_name: str) -> Optional[str]:
        """Get station code from station name"""
        return self.station_codes.get(station_name.lower())

    def get_station_name(self, code: str) -> str:
        """Get proper station name from code"""
        for name, station_code in self.station_codes.items():
            if station_code == code:
                # Clean up names with parentheses
                if '(' in name:
                    return name.split('(')[0].strip()
                return name
        return code  # Fallback to code if name not found

    def generate_station_suggestions(self, partial_name: str, route: str = None) -> List[str]:
        """Generate station suggestions based on input"""
        partial = partial_name.lower()
        matches = []

        # If we have a route, prioritize stations on that route
        if route and route in self.station_sequences:
            station_codes = self.station_sequences[route]
            for code in station_codes:
                name = self.get_station_name(code)
                if name and partial in name.lower():
                    matches.append(name)

        # Also check all stations
        for name in self.station_codes.keys():
            if partial in name.lower() and name not in matches:
                matches.append(name)

        return sorted(matches, key=lambda x: len(x))[:5]  # Return top 5 shortest matches

    def parse_relative_date(self, date_str: str) -> Optional[str]:
        """Improved relative date parsing"""
        today = datetime.now().date()
        date_str = date_str.lower().strip()

        if date_str in ['today', 'now']:
            return today.strftime('%d/%m/%Y')
        elif date_str == 'tomorrow':
            return (today + timedelta(days=1)).strftime('%d/%m/%Y')
        elif date_str == 'day after tomorrow':
            return (today + timedelta(days=2)).strftime('%d/%m/%Y')
        elif date_str.startswith('next '):
            if date_str.endswith('week'):
                return (today + timedelta(weeks=1)).strftime('%d/%m/%Y')
            elif date_str.endswith('month'):
                return (today.replace(month=today.month + 1)).strftime('%d/%m/%Y')
        elif date_str.startswith('in '):
            parts = date_str.split()
            if len(parts) == 3 and parts[1].isdigit():
                num = int(parts[1])
                if parts[2] in ['day', 'days']:
                    return (today + timedelta(days=num)).strftime('%d/%m/%Y')
                elif parts[2] in ['week', 'weeks']:
                    return (today + timedelta(weeks=num)).strftime('%d/%m/%Y')
        elif date_str.startswith('on '):
            day_name = date_str[3:].capitalize()
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']
            if day_name in weekdays:
                days_ahead = (weekdays.index(day_name) - today.weekday()) % 7
                if days_ahead == 0:  # Today is that day
                    days_ahead = 7  # Next week
                return (today + timedelta(days=days_ahead)).strftime('%d/%m/%Y')

        # Try standard date formats
        try:
            datetime.strptime(date_str, '%d/%m/%Y')
            return date_str
        except ValueError:
            pass

        return None

    def parse_time(self, time_str: str) -> Optional[str]:
        """Parse various time formats into HH:MM"""
        time_str = time_str.lower().strip()

        if time_str == "noon":
            return "12:00"
        if time_str == "midnight":
            return "00:00"

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
        print(f"\nExtracting info from: '{text}'")
        extracted = {}
        missing = []

        # Extract stations first (your existing code)
        from_match = re.search(
            r'(?:from|departing)\s+([a-z\s]+?)\s+(?:to|going to|arriving in|arriving at)\s+([a-z\s]+?)(?:\s+(?:on|at)\s+|$)',
            text, re.IGNORECASE)
        if from_match:
            from_station = self.match_station(from_match.group(1))
            to_station = self.match_station(from_match.group(2))
            if from_station and to_station:
                extracted['origin'] = self.station_codes[from_station.lower()]
                extracted['destination'] = self.station_codes[to_station.lower()]

        # if 'return' in text.lower():
        #     self.booking_details['is_return'] = True
        #     try:
        #         # Extract return date/time if mentioned
        #         return_text = text.split('return', 1)[1]  # Get text after "return"
        #         dt = parse(return_text, fuzzy=True, dayfirst=True)
        #         extracted['return_date'] = dt.strftime('%d/%m/%Y')
        #         extracted['return_time'] = dt.strftime('%H:%M')
        #     except:
        #         pass

        # NEW: Simplified date/time extraction using dateutil
        try:
            dt = parse(text, fuzzy=True, dayfirst=True)
            extracted['date'] = dt.strftime('%d/%m/%Y')
            extracted['time'] = dt.strftime('%H:%M')
        except:
            pass  # Date/time remains None if parsing fails

        # Set defaults
        extracted.setdefault('adults', '1')
        extracted.setdefault('children', '0')

        # Check required fields
        # if self.booking_details.get('is_return'):
        #     required.extend(['return_date', 'return_time'])
        required = ['origin', 'destination', 'date', 'time']
        missing = [field for field in required if field not in extracted]

        print(f"Extracted: {extracted}")
        print(f"Missing: {missing}")
        return extracted, missing

    def is_station_on_route(self, station_code: str, route: str) -> bool:
        """Check if station exists in route sequence"""
        return station_code in self.station_sequences.get(route, [])

    def validate_station_on_route(self, station_name: str, route: str) -> Tuple[bool, str]:
        """Check if station exists and is on the specified route"""
        matched_station = self.match_station(station_name)
        if not matched_station:
            return False, "Station not recognized"

        station_code = self.station_codes.get(matched_station.lower())
        if not station_code:
            return False, "Station code not found"

        # Special case for London Liverpool Street
        if 'london' in matched_station.lower() and 'LST' in self.station_sequences.get(route, []):
            return True, 'LST'

        if station_code not in self.station_sequences.get(route, []):
            route_stations = [self.get_station_name(code) for code in self.station_sequences[route]]
            return False, f"This station isn't on your {route} route. Please choose from: {', '.join(route_stations)}"

        return True, station_code

    def generate_missing_prompt(self, missing_fields: List[str]) -> str:
        """Generate a prompt asking for missing information"""
        field_names = {
            'origin': "departure station",
            'destination': "destination station",
            'date': "travel date",
            'time': "travel time"
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

    def detect_route(self, text):
        text = text.lower()
        if "norwich" in text and "london" in text:
            if text.find("norwich") < text.find("london"):
                return 'Norwich-LST'
            return 'LST-Norwich'
        return None

    def validate_direction(self, route: str, current: str, destination: str) -> bool:
        """Properly validate travel direction based on actual sequences"""
        try:
            stations = self.station_sequences[route]
            return stations.index(current) < stations.index(destination)
        except (ValueError, KeyError):
            return False

    def detect_intent(self, text: str) -> str:
        processed = self.preprocess_text(text)
        if not processed:
            return "unknown"

        tokens = set(processed)

        # Enhanced delay detection
        delay_triggers = {
            'delay', 'late', 'arriv', 'time', 'status',
            'long', 'when', 'current', 'predict', 'hold',
            'behind', 'schedule', 'timetable', 'due'
        }

        if tokens & delay_triggers:
            # Check for route using processed tokens
            route = None
            if 'norwich' in tokens and 'london' in tokens:
                norwich_pos = processed.index('norwich') if 'norwich' in processed else -1
                london_pos = processed.index('london') if 'london' in processed else -1

                if norwich_pos >= 0 and london_pos >= 0:
                    route = 'Norwich-LST' if norwich_pos < london_pos else 'LST-Norwich'

            # Check for delay duration with better pattern matching
            delay_mins = None
            for i, token in enumerate(processed):
                if token.isdigit():
                    # Check next token for time unit
                    if i + 1 < len(processed) and processed[i + 1] in ['min', 'minut', 'hour', 'hr']:
                        delay_mins = int(token)
                        if processed[i + 1] in ['hour', 'hr']:
                            delay_mins *= 60
                        break

            if route or delay_mins:
                return "complex_delay_inquiry"

        # Greeting detection
        greeting_words = {'hello', 'hi', 'hey', 'greet'}
        if tokens & greeting_words:
            return "greeting"

        # Farewell detection
        farewell_words = {'bye', 'goodbye', 'exit', 'quit', 'farewell'}
        if tokens & farewell_words:
            return "farewell"

        # Booking detection
        booking_triggers = {
            'book', 'ticket', 'journey', 'travel', 'train',
            'schedul', 'timet', 'from', 'to', 'depart', 'arriv',
            'go', 'need', 'want', 'would'  # Stems/lemmas of common words
        }
        if tokens & booking_triggers:
            return "booking"

        # Delay detection
        delay_triggers = {
            'delay', 'late', 'arriv', 'time', 'status',
            'long', 'when', 'current', 'predict'
        }
        if tokens & delay_triggers:
            return "delay_inquiry"

        return "unknown"

    def extract_delay_info(self, text: str) -> dict:
        """Extract delay-related information from text"""
        extracted = {}
        text_lower = text.lower()

        # Check for route
        if 'norwich' in text_lower and 'london' in text_lower:
            if text_lower.find('norwich') < text_lower.find('london'):
                extracted['route'] = 'Norwich-LST'
            else:
                extracted['route'] = 'LST-Norwich'

        # Check for delay duration
        delay_match = re.search(r'(\d+)\s*(minute|min|hour|hr)s?', text_lower)
        if delay_match:
            extracted['delay'] = int(delay_match.group(1))
            # Convert hours to minutes if needed
            if delay_match.group(2) in ['hour', 'hr']:
                extracted['delay'] *= 60

        # Check for station mentions
        for word in text_lower.split():
            matched_station = self.match_station(word)
            if matched_station:
                # First station mentioned is likely current station
                if 'current_station' not in extracted:
                    extracted['current_station'] = matched_station
                # Second station mentioned is likely destination
                elif 'destination' not in extracted:
                    extracted['destination'] = matched_station

        return extracted

    def make_delay_prediction(self) -> str:
        """Generate prediction when all details are available"""
        try:
            route = self.delay_details['route']
            current = self.delay_details['current_station']
            dest = self.delay_details['destination']
            delay = self.delay_details['reported_delay']

            if not self.validate_direction(route, current, dest):
                return ("You appear to be traveling in the wrong direction for this route. "
                        f"From {self.get_station_name(current)}, you should be heading towards: "
                        f"{', '.join(self.get_station_name(code) for code in self.station_sequences[route][self.station_sequences[route].index(current) + 1:])}")

            prediction = self.predict_arrival(route, current, delay, dest)
            self.current_state = self.STATES['GET_OPTION']
            return f"{prediction}\n\n{self.options_prompt}"

        except Exception as e:
            return f"Sorry, I couldn't make a prediction: {str(e)}"

    def handle_destination_input(self, user_input: str) -> str:
        is_valid, message = self.validate_station_on_route(
            user_input,
            self.delay_details['route']
        )

        if not is_valid:
            # Generate suggestions from route stations
            route_stations = self.station_sequences.get(self.delay_details['route'], [])
            suggestions = [
                self.get_station_name(code)
                for code in route_stations
                if user_input.lower() in self.get_station_name(code).lower()
            ]

            if suggestions:
                return f"{message}\nDid you mean: {', '.join(suggestions[:3])}"
            return message

        self.delay_details['destination'] = message  # message contains station_code here
        return self.make_delay_prediction()

    def handle_delay_inquiry(self, user_input: str) -> str:
        """Handle complex delay inquiries with proper validation"""
        # Extract route
        if 'norwich' in user_input.lower() and 'london' in user_input.lower():
            if user_input.lower().find('norwich') < user_input.lower().find('london'):
                route = 'Norwich-LST'
            else:
                route = 'LST-Norwich'
        else:
            return "Please specify your route as 'Norwich to London' or 'London to Norwich'"

        # Extract delay minutes
        delay_match = re.search(r'(\d+)\s*(minute|min|hour|hr)', user_input.lower())
        delay = int(delay_match.group(1)) if delay_match else None

        # Update context
        self.delay_details = {'route': route}
        if delay:
            self.delay_details['reported_delay'] = delay

        # Determine next question
        if not self.delay_details.get('current_station'):
            self.current_state = self.STATES['GET_CURRENT_STATION']
            return "Which station are you currently at?"
        elif not self.delay_details.get('reported_delay'):
            self.current_state = self.STATES['GET_DELAY_DURATION']
            return "How many minutes delayed is your train?"
        elif not self.delay_details.get('destination'):
            self.current_state = self.STATES['GET_DESTINATION']
            return "Which station is your destination?"
        else:
            return self.make_delay_prediction()

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
            if intent == "delay_inquiry":
                self.current_state = self.STATES['DELAY_INQUIRY']
                return "I can help with delay predictions. First, which route are you on? (Norwich to London or London to Norwich)"
            else:
                self.current_state = self.STATES['GET_OPTION']
                return f"{random.choice(self.greetings)}\n{self.options_prompt}"

        elif self.current_state == self.STATES['GET_OPTION']:
            if intent == "delay_inquiry" or "2" in user_input:
                self.current_state = self.STATES['DELAY_INQUIRY']
                return "I can help with delay predictions. First, which route are you on? (Norwich to London or London to Norwich)"
            elif intent == "booking" or "1" in user_input:
                self.current_state = self.STATES['PARSE_BOOKING']
                return "Please tell me about your journey. For example: 'I want to travel from Norwich to London tomorrow at 3pm'"
            elif "3" in user_input or intent == "farewell":
                self.current_state = self.STATES['FAREWELL']
                return random.choice(self.farewells)
            else:
                return "I didn't understand that. Please choose:\n1. Book tickets\n2. Check for delays\n3. Exit"

                """Handle complex delay inquiries"""
        intent = self.detect_intent(user_input)

        if intent == "complex_delay_inquiry":
            # Extract available information first
            extracted = self.extract_delay_info(user_input)

            # Store what we found
            if extracted.get('route'):
                self.delay_details['route'] = extracted['route']
            if extracted.get('delay'):
                self.delay_details['reported_delay'] = extracted['delay']

            # Determine what's missing
            missing = []
            if not self.delay_details.get('route'):
                missing.append('route')
            if not self.delay_details.get('current_station'):
                missing.append('current_station')
            if not self.delay_details.get('destination'):
                missing.append('destination')
            if not self.delay_details.get('reported_delay'):
                missing.append('delay_duration')

            # Ask for missing information
            if missing:
                prompts = {
                    'route': "Which route are you on? (Norwich to London or London to Norwich)",
                    'current_station': "Which station are you currently at?",
                    'destination': "Which station is your destination?",
                    'delay_duration': "How many minutes delayed is your train?"
                }

                # Prioritize questions - get route first if missing
                if 'route' in missing:
                    self.current_state = self.STATES['DELAY_INQUIRY']
                    return prompts['route']

                # Then current station
                if 'current_station' in missing:
                    self.current_state = self.STATES['GET_CURRENT_STATION']
                    return prompts['current_station']

                # Then delay duration
                if 'delay_duration' in missing:
                    self.current_state = self.STATES['GET_DELAY_DURATION']
                    return prompts['delay_duration']

                # Finally destination
                if 'destination' in missing:
                    self.current_state = self.STATES['GET_DESTINATION']
                    return prompts['destination']

            # If we have everything, make prediction
            else:
                return self.make_delay_prediction()


        elif self.current_state == self.STATES['DELAY_INQUIRY']:
            # Check for route specification
            if "norwich" in user_input.lower() and "london" in user_input.lower():
                if "to" in user_input.lower() and user_input.lower().index("norwich") < user_input.lower().index(
                        "london"):
                    self.delay_details = {'route': 'Norwich-LST'}
                else:
                    self.delay_details = {'route': 'LST-Norwich'}

                self.current_state = self.STATES['GET_CURRENT_STATION']
                return "Which station are you currently at?"
            return "Please specify your route as 'Norwich to London' or 'London to Norwich'"

        elif self.current_state == self.STATES['ASK_RETURN']:
            if user_input.lower() in ['yes', 'y']:
                self.booking_details['is_return'] = True
                self.current_state = self.STATES['GET_MISSING_INFO']
                return "Please provide your return date and time (e.g. 'June 6th at 4pm')"
            else:
                self.booking_details['is_return'] = False
                self.current_state = self.STATES['CONFIRM_BOOKING']
                return self.process_booking()  # Proceed with one-way ticket

        elif self.current_state == self.STATES['GET_CURRENT_STATION']:
            matched_station = self.match_station(user_input)
            if matched_station:
                self.delay_details['current_station'] = self.station_codes.get(
                    matched_station.lower())  # Store station code
                self.current_state = self.STATES['GET_DELAY_DURATION']
                return "How many minutes delayed?"
            return "Station not recognized. Please try again."


        elif self.current_state == self.STATES['GET_DELAY_DURATION']:
            match = re.search(r'(\d+)', user_input)
            if match:
                self.delay_details['reported_delay'] = int(match.group(1))
                self.current_state = self.STATES['GET_DESTINATION']
                return "Which station is your destination?"
            return "Please specify the delay in minutes (e.g., '15 minutes')."


        elif self.current_state == self.STATES['PARSE_BOOKING']:
            extracted, missing = self.extract_booking_info(user_input)
            if extracted:
                self.booking_details.update(extracted)
                self.missing_fields = missing

                if not missing:
                    self.booking_details['confirmed'] = False
                    self.current_state = self.STATES['GET_MISSING_INFO']
                    return self.process_booking()  # Triggers confirmation message
                else:
                    self.current_state = self.STATES['GET_MISSING_INFO']
                    response = "I understood:"
                    if 'origin' in extracted:
                        response += f"\n- From: {self.get_station_name(extracted['origin'])}"
                    if 'destination' in extracted:
                        response += f"\n- To: {self.get_station_name(extracted['destination'])}"
                    if 'date' in extracted:
                        response += f"\n- Date: {extracted['date']}"
                    if 'time' in extracted:
                        response += f"\n- Time: {extracted['time']}"
                    response += "\n\n" + self.generate_missing_prompt(missing)
                    return response
            return ("I couldn't understand your booking request. "
                    "Please try something like: 'I want to travel from Norwich to London tomorrow at 3pm'")

        elif self.current_state == self.STATES['GET_MISSING_INFO']:
            # Handle confirmation separately
            if user_input.lower() in ['yes', 'y']:
                self.booking_details['confirmed'] = True
                return self.process_booking(confirm=True)

            # Only extract info if it's not a simple confirmation
            extracted, missing = self.extract_booking_info(user_input)
            if extracted:
                self.booking_details.update(extracted)
                self.missing_fields = missing
                if not missing:
                    self.booking_details['confirmed'] = False
                    return self.process_booking()
                else:
                    response = "I understood:"
                    if 'origin' in extracted:
                        response += f"\n- From: {self.get_station_name(extracted['origin'])}"
                    if 'destination' in extracted:
                        response += f"\n- To: {self.get_station_name(extracted['destination'])}"
                    if 'date' in extracted:
                        response += f"\n- Date: {extracted['date']}"
                    if 'time' in extracted:
                        response += f"\n- Time: {extracted['time']}"
                    response += "\n\n" + self.generate_missing_prompt(missing)
                    return response
            return "Please provide the missing information: " + self.generate_missing_prompt(self.missing_fields)

        elif self.current_state == self.STATES['GET_DESTINATION']:
            response = self.handle_destination_input(user_input)
            if "Normally this journey" in response:  # Successful prediction
                self.current_state = self.STATES['GET_OPTION']
            return response

    def process_booking(self, confirm=False):
        summary = "Here's your journey summary:\n"
        summary += f"From: {self.get_station_name(self.booking_details['origin'])}\n"
        summary += f"To: {self.get_station_name(self.booking_details['destination'])}\n"
        summary += f"Date: {self.booking_details['date']}\n"
        summary += f"Time: {self.booking_details['time']}\n"
        summary += f"Adults: {self.booking_details['adults']}\n"
        summary += f"Children: {self.booking_details['children']}\n"

        # Add return info if available
        if self.booking_details.get('is_return'):
            if self.booking_details.get('return_date'):
                summary += f"Return Date: {self.booking_details['return_date']}\n"
            if self.booking_details.get('return_time'):
                summary += f"Return Time: {self.booking_details['return_time']}\n"

        if confirm:
            try:
                result = compare_ticket_prices(
                    origin=self.booking_details['origin'],
                    destination=self.booking_details['destination'],
                    depart_date=self.booking_details['date'],
                    depart_time=self.booking_details['time'],
                    depart_type="departing",  # later you can let the user choose
                    is_return=self.booking_details.get('is_return', False),
                    return_date=self.booking_details.get('return_date'),
                    return_time=self.booking_details.get('return_time'),
                    return_type="departing",  # later you can let the user choose
                    adults=self.booking_details['adults'],
                    children=self.booking_details['children']
                )

                result_summary = "\n📊 Ticket Comparison Result:\n"

                if result['national_rail']['total_price']:
                    result_summary += (
                        f"\n🚆 National Rail: £{result['national_rail']['total_price']:.2f} "
                        f"| {result['national_rail']['url']}"
                    )
                else:
                    result_summary += "\n❌ National Rail could not find a price."

                if result['thameslink']['total_price']:
                    result_summary += (
                        f"\n🚆 Thameslink: £{result['thameslink']['total_price']:.2f} "
                        f"| {result['thameslink']['url']}"
                    )
                else:
                    result_summary += "\n❌ Thameslink could not find a price."

                result_summary += f"\n\n🧠 Verdict: {result['verdict']}"

                self.current_state = self.STATES['FAREWELL']
                return result_summary

            except Exception as e:
                return f"❌ Error during ticket search: {str(e)}"

        else:
            # Ask for confirmation
            if not self.booking_details.get('is_return'):
                self.current_state = self.STATES['ASK_RETURN']
                return summary + "\nWould you like to book a return ticket? (yes/no)"

            return summary + "\nWould you like me to find the best tickets for this journey? (yes/no)"

    def calculate_normal_time(self, route: str, start: str, end: str) -> float:
        """Calculate normal travel time between stations"""
        try:
            stations = self.station_sequences[route]
            start_idx = stations.index(start)
            end_idx = stations.index(end)

            # Get typical times between each pair
            total = 0
            for i in range(start_idx, end_idx):
                from_stn = stations[i]
                to_stn = stations[i + 1]
                pair = (from_stn, to_stn)

                if pair in self.delay_stats:
                    total += self.delay_stats[pair]['median_delay'] + 10  # Add typical travel time
                else:
                    total += 15  # Default time between stations

            return total
        except (ValueError, KeyError):
            return None

    def calculate_normal_travel_times(self, route):
        """Calculate normal travel times between stations for a route"""
        if route not in self.delay_data or not isinstance(self.delay_data[route], pd.DataFrame):
            return None

        df = self.delay_data[route].copy()
        normal_times = {}

        # First try to automatically detect time columns
        time_cols = {'arrival': None, 'departure': None}

        # Common column name patterns to check
        arrival_patterns = ['arrival', 'pta', 'planned_arrival']
        departure_patterns = ['departure', 'ptd', 'planned_departure']

        # Try to find matching columns
        for col in df.columns:
            col_lower = col.lower()
            if not time_cols['arrival'] and any(pattern in col_lower for pattern in arrival_patterns):
                time_cols['arrival'] = col
            if not time_cols['departure'] and any(pattern in col_lower for pattern in departure_patterns):
                time_cols['departure'] = col

        # If we couldn't find columns, use hardcoded defaults based on route
        if not time_cols['arrival'] or not time_cols['departure']:
            if route == 'Norwich-LST':
                time_cols = {
                    'arrival': 'planned_arrival_time',
                    'departure': 'planned_departure_time'
                }
            else:  # LST-Norwich
                time_cols = {
                    'arrival': 'gbtt_pta',
                    'departure': 'gbtt_ptd'
                }

        # Check if the required columns exist
        missing_cols = [col for col in time_cols.values() if col not in df.columns]
        if missing_cols:
            print(f"Warning: Could not find time columns in {route} data. Tried: {time_cols}")
            print(f"Actual columns: {df.columns.tolist()}")
            return None

        # Convert time columns to datetime.time objects
        for col in time_cols.values():
            # First convert to string if needed
            df[col] = df[col].astype(str)
            # Then parse times, handling empty/missing values
            df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce').dt.time

        stations = self.station_sequences[route]

        for i in range(len(stations) - 1):
            from_stn = stations[i]
            to_stn = stations[i + 1]

            # Get all services between these stations
            from_df = df[df['location'] == from_stn]
            to_df = df[df['location'] == to_stn]

            # Merge on rid and date_of_service to match the same train
            merged = pd.merge(
                from_df[['rid', 'date_of_service', time_cols['departure']]].rename(
                    columns={time_cols['departure']: 'departure_time'}),
                to_df[['rid', 'date_of_service', time_cols['arrival']]].rename(
                    columns={time_cols['arrival']: 'arrival_time'}),
                on=['rid', 'date_of_service'],
                suffixes=('_from', '_to')
            )

            if not merged.empty:
                # Calculate travel time in minutes, handling NaT values
                def calculate_time(row):
                    try:
                        dep_time = row['departure_time']
                        arr_time = row['arrival_time']

                        if pd.isna(dep_time) or pd.isna(arr_time):
                            return np.nan

                        # Create datetime objects for calculation
                        dep_dt = datetime.combine(datetime.today(), dep_time)
                        arr_dt = datetime.combine(datetime.today(), arr_time)

                        # Handle overnight trains (negative time difference)
                        if arr_dt < dep_dt:
                            arr_dt += timedelta(days=1)

                        return (arr_dt - dep_dt).total_seconds() / 60
                    except Exception as e:
                        print(f"Error calculating time: {e}")
                        return np.nan

                merged['travel_time'] = merged.apply(calculate_time, axis=1)

                # Drop rows with invalid times
                merged = merged.dropna(subset=['travel_time'])

                if not merged.empty:
                    # Store median travel time
                    normal_times[(from_stn, to_stn)] = merged['travel_time'].median()
                    print(f"Normal time {from_stn}->{to_stn}: {normal_times[(from_stn, to_stn)]:.1f} minutes")
                else:
                    print(f"No valid time data for {from_stn}->{to_stn}")
                    normal_times[(from_stn, to_stn)] = 30  # Default fallback
            else:
                print(f"No matching services for {from_stn}->{to_stn}")
                normal_times[(from_stn, to_stn)] = 30  # Default fallback

        return normal_times

    def predict_arrival(self, route, current_station, delay, destination):
        """Predict arrival time with fallback behavior"""
        normal_times = self.normal_travel_times.get(route, {})
        stations = self.station_sequences.get(route, [])

        if not normal_times:
            print(f"Warning: No normal travel times available for route {route}")
            # Create default normal times
            normal_times = {(stations[i], stations[i + 1]): 30
                            for i in range(len(stations) - 1)}

        if current_station not in stations or destination not in stations:
            print(f"Warning: Stations not in route {route}")
            return None

        current_idx = stations.index(current_station)
        dest_idx = stations.index(destination)

        # Calculate normal travel time
        normal_travel_time = 0
        for i in range(current_idx, dest_idx):
            from_stn = stations[i]
            to_stn = stations[i + 1]
            pair = (from_stn, to_stn)
            normal_travel_time += normal_times.get(pair, 30)  # Default to 30 minutes if no data

        # Add the reported delay
        total_time = normal_travel_time + delay

        # Get current time
        now = datetime.now()

        # Calculate estimated arrival time
        estimated_arrival = now + timedelta(minutes=total_time)

        # Format the output
        output = (
            f"Normally this journey takes {normal_travel_time:.1f} minutes.\n"
            f"With your {delay} minute delay, the total expected travel time is {total_time:.1f} minutes.\n"
            f"Estimated arrival time: {estimated_arrival.strftime('%H:%M')}"
        )

        return output

    def calculate_schedule_times(self, route, stations):
        """Calculate typical schedule times between stations"""
        if route not in self.delay_data or not isinstance(self.delay_data[route], pd.DataFrame):
            return None

        df = self.delay_data[route]
        schedule_times = [0]  # Start at 0 minutes for first station

        for i in range(1, len(stations)):
            from_stn = stations[i - 1]
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

        # if return_date_str and return_time_str:
        #     r_day, r_month, r_year = return_date_str.split("/")
        #     r_hour, r_minute = return_time_str.split(":")
        #     r_minute = self.round_down_to_quarter(r_minute)
        #     return_date = f"{r_day}{r_month}{r_year[2:]}"
        #     return_hour = r_hour.zfill(2)

        #     params.update({
        #         "type": "return",
        #         "returnType": "departing",
        #         "returnDate": return_date,
        #         "returnHour": return_hour,
        #         "returnMin": r_minute
        #     })
        # else:
        #     params["type"] = "single"

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
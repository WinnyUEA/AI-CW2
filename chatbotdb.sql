DROP TABLE IF EXISTS stations_typed;

-- Stores user info who interacted with the chatbot
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stores conversation messages between user and chatbot
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    message TEXT NOT NULL,
    sender_type VARCHAR(10) CHECK (sender_type IN ('user', 'bot')),
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    account_type VARCHAR(10) CHECK (account_type IN ('user', 'staff')) NOT NULL
);


-- Stores UK train station data
CREATE TABLE stations(
    station_code VARCHAR(10) PRIMARY KEY,
    station_name VARCHAR(100) NOT NULL,
    full_name TEXT,
    alias VARCHAR(50),
    tiploc_code VARCHAR(10),
    code TEXT
);
SELECT * FROM stations;

-- Logical routes between two stations 
CREATE TABLE train_routes (
    id SERIAL PRIMARY KEY,
    from_station_code VARCHAR(10) REFERENCES stations(station_code),
    to_station_code VARCHAR(10) REFERENCES stations(station_code),
    train_operator VARCHAR(100) NOT NULL
);


-- Real recorded data from past train journeys
CREATE TABLE train_history (
    id SERIAL PRIMARY KEY,
    train_number VARCHAR(20) NOT NULL,
    route_id INT REFERENCES train_routes(id),
    run_date DATE NOT NULL,
    dep_scheduled TIME,
    dep_actual TIME,
    arr_scheduled TIME,
    arr_actual TIME,
    total_delay_mins INT
);


-- Predicts the train time which will be done by the ML model
CREATE TABLE delay_predictions (
    id SERIAL PRIMARY KEY,
    train_number VARCHAR(20),
    current_station_code VARCHAR(10) REFERENCES stations(station_code),
    destination_code VARCHAR(10) REFERENCES stations(station_code),
    estimated_delay_mins INT,
    predicted_arrival_time TIME,
    prediction_model VARCHAR(100),
    generated_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Table for backup plans or emergencies
CREATE TABLE contingency_guidelines (
    id SERIAL PRIMARY KEY,
    issue_type VARCHAR(100),
    description TEXT,
    action_plan TEXT,
    support_contact TEXT
);

CREATE TABLE staff (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    full_name TEXT
);
INSERT INTO staff (username, password, full_name)
VALUES 
('aldrich', 'aldrich123', 'Aldrich Monteiro'),
('prasid', 'prasid123', 'Prasid Shreshta'),
('Winny', 'yeyin123', 'Ye Yin');

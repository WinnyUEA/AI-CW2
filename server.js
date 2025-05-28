const express = require('express');
const session = require('express-session');
const app = express();
const bcrypt = require('bcrypt');
const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch'); 
const { Pool } = require('pg');

const kbPath = path.join(__dirname, 'data', 'knowledge_base.json');

// This is for db setup using pool
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres',
  password: 'admin',
  port: 5432,
});

// these are the middlewares
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// code for session management
app.use(session({
  secret: 'your_secret_key',
  resave: false,
  saveUninitialized: false,
}));

// these are the static fils
app.use('/static', express.static(path.join(__dirname, 'static')));
app.use(express.static(path.join(__dirname, 'public')));

// Routes for file handling
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'login.html'));
});

app.get('/signup', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'signup.html'));
});

// the logout route
app.get('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) {
      console.error(err);
      return res.status(500).send("Logout failed");
    }
    res.redirect('/login');
  });
});

// Inserting a new user which is saved in the users colum in the db
async function insertUser(username, password) {
  const hashedPassword = await bcrypt.hash(password, 10);
  const query = `INSERT INTO users (username, password) VALUES ($1, $2) RETURNING *`;
  return pool.query(query, [username, hashedPassword]);
}

// signig up route
app.post('/signup', async (req, res) => {
  const { username, password } = req.body;
  try {
    await insertUser(username, password);
    res.redirect('/login');
  } catch (err) {
    console.error(err);
    res.status(500).send('Signup failed');
  }
});

// this is the login route
app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  try {
    // this is the staff login route which stores password in plain txt
    const staffResult = await pool.query('SELECT * FROM staff WHERE username = $1 AND password = $2', [username, password]);
    if (staffResult.rows.length > 0) {
      req.session.userId = staffResult.rows[0].id;
      req.session.userType = 'staff';
      return res.json({ success: true, userType: 'staff' });
    }

    // this is user login which hashes the password using bcrypt
    const userResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
    if (userResult.rows.length > 0) {
      const match = await bcrypt.compare(password, userResult.rows[0].password);
      if (match) {
        req.session.userId = userResult.rows[0].id;
        req.session.userType = 'user';
        return res.json({ success: true, userType: 'user' });
      }
    }

    res.json({ success: false, message: 'Invalid credentials' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Login error' });
  }
});

//  the chat for the users which calls python chatbot via Flask
app.post('/chat', async (req, res) => {
  if (!req.session.userId) return res.status(401).json({ error: 'Unauthorized' });

  const userMessage = req.body.message;

  try {
    const flaskRes = await fetch('http://localhost:5001/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage })
    });

    const data = await flaskRes.json();
    const botReply = data.reply;

    await pool.query(
      'INSERT INTO chat_history (user_id, message, sender_type, sent_at) VALUES ($1, $2, $3, NOW())',
      [req.session.userId, userMessage, 'user']
    );
    await pool.query(
      'INSERT INTO chat_history (user_id, message, sender_type, sent_at) VALUES ($1, $2, $3, NOW())',
      [req.session.userId, botReply, 'bot']
    );

    res.json({ reply: botReply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Chatbot error' });
  }
});

//  the chat for the staff which calls python chatbot via Flask
app.post('/staff-chat', async (req, res) => {
  if (req.session.userType !== 'staff') {
    return res.status(403).json({ reply: " Unauthorized access. Staff only." });
  }

  const userMessage = req.body.message;

  try {
    const flaskRes = await fetch('http://localhost:5002/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage })
    });

    const data = await flaskRes.json();
    res.json({ reply: data.reply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ reply: ' Error communicating with staff bot.' });
  }
});

// checks session who is logged in wether user or staff
app.get('/session-status', (req, res) => {
  res.json({
    loggedIn: !!req.session.userId,
    userType: req.session.userType || null
  });
});

// fetches and stores all of the chat history in the database
app.get('/chat-history', async (req, res) => {
  if (!req.session.userId) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const result = await pool.query(
      'SELECT message, sender_type, sent_at FROM chat_history WHERE user_id = $1 ORDER BY sent_at ASC',
      [req.session.userId]
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load chat history' });
  }
});

//the server starts now
app.listen(3000, () => {
  console.log(' The Server is running at http://localhost:3000');
});

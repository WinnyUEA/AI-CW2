const express = require('express');
const session = require('express-session');
const app = express();
const bcrypt = require('bcrypt');
const path = require('path');
const { Pool } = require('pg');

// PostgreSQL setup
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'postgres',
  password: 'admin',
  port: 5432,
});

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Sessions
app.use(session({
  secret: 'your_secret_key', // use a strong random string in production
  resave: false,
  saveUninitialized: false,
}));

// Static files
app.use('/static', express.static(path.join(__dirname, 'static')));

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

app.get('/login', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'login.html'));
});

app.get('/login.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'login.html'));
});

app.get('/signup', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'signup.html'));
});

app.get('/signup.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'signup.html'));
});

// Logout
app.get('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) {
      console.error(err);
      return res.status(500).send("Logout failed");
    }
    res.redirect('/login');
  });
});

// Insert user
async function insertUser(username, password) {
  const hashedPassword = await bcrypt.hash(password, 10);
  const query = `INSERT INTO users (username, password) VALUES ($1, $2) RETURNING *`;
  return pool.query(query, [username, hashedPassword]);
}

// Signup
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

// Login (✅ Correct handling: bcrypt for users, plain text for staff)
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    // 1. Check USERS table (hashed passwords)
    const userRes = await pool.query("SELECT * FROM users WHERE username = $1", [username]);
    if (userRes.rows.length > 0) {
      const isMatch = await bcrypt.compare(password, userRes.rows[0].password);
      if (isMatch) {
        req.session.userId = userRes.rows[0].id;
        req.session.userType = 'user'; // normal user
        return res.json({ success: true, userType: 'user' });
      }
    }

    // 2. Check STAFF table (plain text passwords)
    const staffRes = await pool.query("SELECT * FROM staff WHERE username = $1", [username]);
    if (staffRes.rows.length > 0) {
      const staff = staffRes.rows[0];
      if (password === staff.password) {
        req.session.userId = staff.id;
        req.session.userType = 'staff'; // staff user
        return res.json({ success: true, userType: 'staff' });
      }
    }

    // 3. No match
    res.status(401).json({ success: false, message: "Invalid username or password" });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// Chat message save route
app.post('/chat', async (req, res) => {
  if (!req.session.userId) return res.status(401).json({ error: 'Unauthorized' });

  const userId = req.session.userId;
  const userMessage = req.body.message;

  try {
    // Save user message
    await pool.query(
      'INSERT INTO chat_history (user_id, message, sender_type, sent_at) VALUES ($1, $2, $3, NOW())',
      [userId, userMessage, 'user']
    );

    // Simulated bot reply (replace with real logic later)
    const botReply = "This is a bot reply";

    // Save bot reply
    await pool.query(
      'INSERT INTO chat_history (user_id, message, sender_type, sent_at) VALUES ($1, $2, $3, NOW())',
      [userId, botReply, 'bot']
    );

    res.json({ reply: botReply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Chat saving failed' });
  }
});

app.post('/staff-chat', async (req, res) => {
  const { message } = req.body;
  const userId = req.session.userId;

  const reply = `🚧 Staff Bot says: "${message}" understood. We're dispatching rail support.`;

  await pool.query("INSERT INTO chat_history (user_id, message, sender_type) VALUES ($1, $2, 'user')", [userId, message]);
  await pool.query("INSERT INTO chat_history (user_id, message, sender_type) VALUES ($1, $2, 'bot')", [userId, reply]);

  res.json({ reply });
});

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

// Check session login status
app.get('/session-status', (req, res) => {
  if (req.session.userId) {
    res.json({ loggedIn: true, userType: req.session.userType });
  } else {
    res.json({ loggedIn: false });
  }
});

// Start server
app.listen(3000, () => {
  console.log('✅ Server running at http://localhost:3000');
});

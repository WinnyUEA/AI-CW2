const express = require('express');
const session = require('express-session');
const app = express();
const bcrypt = require('bcrypt');
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');

const kbPath = path.join(__dirname, 'data', 'knowledge_base.json');

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
  secret: 'your_secret_key',
  resave: false,
  saveUninitialized: false,
}));

// Static files
app.use('/static', express.static(path.join(__dirname, 'static')));
app.use(express.static(path.join(__dirname, 'public')));

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

// Login (bcrypt for users, plain text for staff)
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    // Staff login (plain text)
    const staffResult = await pool.query('SELECT * FROM staff WHERE username = $1 AND password = $2', [username, password]);
    if (staffResult.rows.length > 0) {
      req.session.userId = staffResult.rows[0].id;
      req.session.userType = 'staff';
      return res.json({ success: true, userType: 'staff' });
    }

    // User login (bcrypt)
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

// User chat
app.post('/chat', async (req, res) => {
  if (!req.session.userId) return res.status(401).json({ error: 'Unauthorized' });

  const userId = req.session.userId;
  const userMessage = req.body.message;

  try {
    await pool.query(
      'INSERT INTO chat_history (user_id, message, sender_type, sent_at) VALUES ($1, $2, $3, NOW())',
      [userId, userMessage, 'user']
    );

    const botReply = "This is a bot reply";

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

// Staff Chat
app.post('/staff-chat', (req, res) => {
  if (req.session.userType !== 'staff') {
    return res.status(403).json({ reply: "❌ Unauthorized access. Staff only." });
  }

  const message = req.body.message.trim().toLowerCase();

  if (message === 'reset') {
    req.session.staffChatState = null;
    return res.json({ reply: "🔄 Conversation reset. Please tell me the location of the blockage." });
  }

  let kb;
  try {
    const kbRaw = fs.readFileSync(kbPath, 'utf-8');
    kb = JSON.parse(kbRaw);
  } catch (err) {
    console.error('Knowledge base load error:', err);
    return res.status(500).json({ reply: "❗ Error loading knowledge base." });
  }

  if (!req.session.staffChatState) {
    req.session.staffChatState = { step: 'location' };
  }

  const chatState = req.session.staffChatState;

  const Fuse = require('fuse.js');
  const fuse = new Fuse(kb, {
    keys: ['location'],
    threshold: 0.3,
    ignoreLocation: true,
    includeScore: true
  });

  // Step 1: Location
  if (chatState.step === 'location') {
    const results = fuse.search(message);

    if (results.length === 0) {
      return res.json({ reply: "❗ I couldn't match that location.\n🗺️ Please enter a valid known location (e.g. 'Stratford - Maryland').\nOr type `reset` to start over." });
    }

    chatState.location = results[0].item.location;
    chatState.step = 'blockage_type';
    return res.json({ reply: `📍 Location noted: **${chatState.location}**\n\n❓ Is the blockage **partial** or **full**?` });
  }

  // Step 2: Blockage Type
  if (chatState.step === 'blockage_type') {
    const type = message.includes('partial') ? 'partial'
                : message.includes('full') ? 'full'
                : null;

    if (!type) {
      return res.json({ reply: "❗ Please reply with either **partial** or **full** for the blockage type.\nOr type `reset` to restart." });
    }

    chatState.blockage_type = type;

    const match = kb.find(entry =>
      entry.location.toLowerCase() === chatState.location.toLowerCase() &&
      entry.blockage_type.toLowerCase() === type
    );

    req.session.staffChatState = null;

    if (match) {
      return res.json({
        reply:
          `✅ Thank you. Here is the contingency advice:\n\n` +
          `📍 **Location**: ${match.location}\n` +
          `🛑 **Blockage Type**: ${match.blockage_type}\n` +
          `📘 **Advice**: ${match.advice}\n` +
          `🚍 **Alternative Transport**: ${match.alt_transport || 'N/A'}\n\n` +
          `📝 **Passenger Notes**: ${match.passenger_notes || 'N/A'}\n` +
          `👷‍♂️ **Staff Notes**: ${match.staff_notes || 'N/A'}`
      });
    } else {
      return res.json({ reply: "❗ I couldn't find a match for that location and blockage type.\n🔁 Please type `reset` to try again." });
    }
  }

  // Catch-all fallback: guide user based on current step
  if (chatState.step === 'location') {
    return res.json({ reply: "❗ Please enter a valid location to continue.\n🗺️ For example: 'Stratford - Maryland'.\nOr type `reset` to start over." });
  }

  if (chatState.step === 'blockage_type') {
    return res.json({ reply: "❗ Please enter **partial** or **full** as the blockage type.\nOr type `reset` to restart the chat." });
  }

  return res.json({ reply: "❓ Something went wrong. Please type `reset` to start over." });
});

// Chat history
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

// Session status
app.get('/session-status', (req, res) => {
  res.json({
    loggedIn: !!req.session.userId,
    userType: req.session.userType || null
  });
});

// Staff chatbot page
app.get('/staff-chatbot', (req, res) => {
  if (req.session.userType !== 'staff') {
    return res.redirect('/login');
  }
  res.sendFile(path.join(__dirname, 'public', 'staff-chat.html'));
});

// Start server
app.listen(3000, () => {
  console.log('✅ Server running at http://localhost:3000');
});

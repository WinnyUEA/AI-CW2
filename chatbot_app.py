from flask import Flask, render_template, request, jsonify
from ChatBotTuesV2 import TrainTicketChatbotWithBooking

app = Flask(__name__)
bot = TrainTicketChatbotWithBooking()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if user_input.lower() in ['exit', 'quit', 'bye']:
        return jsonify(reply="Thank you for chatting with TrainBot. Have a great day!")

    reply = bot.generate_response(user_input)
    return jsonify(reply=reply)

if __name__ == '__main__':
    app.run(debug=True)

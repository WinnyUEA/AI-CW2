from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from staff_chatbot import StaffChatbot


app = Flask(__name__)
chatbot = StaffChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    reply = chatbot.generate_response(user_input)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(port=5002)

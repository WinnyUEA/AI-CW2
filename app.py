from flask import Flask, request, render_template, jsonify
from chatbot_engine import get_advice
import json

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

#Route to fetch the contigency api
@app.route("/api/contingency")
def contingency_api():
    with open("knowledge_base.json", "r") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)

# app.py

from flask import Flask, request, jsonify, render_template
from llama_helper import LlamaChatbot

app = Flask(__name__)
chatbot = LlamaChatbot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = chatbot.generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

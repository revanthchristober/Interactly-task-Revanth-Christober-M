from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.rag_framework import match_candidates, generate_response

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    job_description = data.get('message')
    matched_candidates = match_candidates(job_description)
    response = generate_response(matched_candidates, job_description)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)



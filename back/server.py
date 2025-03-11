from flask import Flask, request, jsonify
from flask_cors import CORS
from using.classify_sentence import classify_sentence

server = Flask(__name__)
CORS(server)

@server.route('/execute', methods=['POST'])
def api_classify_sentence():
    def run_classification(sentence):
        return classify_sentence(sentence)
    data = request.get_json()
    result = run_classification(data['input'])
    return jsonify({'result': result})

server.run(debug=True)
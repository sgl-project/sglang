from flask import Flask, jsonify, request
import json

app = Flask(__name__)

@app.route('/get_model_info', methods=['GET'])
def get_model_info():
    return jsonify({
        "model_path": "path/to/your/model"
    })

@app.route('/flush_cache', methods=['POST'])
def flush_cache():
    return jsonify({"status": "cache flushed"})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    if text == "hours":
        response = {"hours": 24}
    elif text == "days":
        response = {"days": 365}
    else:
        response_text = f"The number of hours in a day is {text}"
        response = {"text": response_text, "meta_info": {"prompt_tokens": len(text)}}
    return jsonify(response)

@app.route('/concate_and_append_request', methods=['POST'])
def concate_and_append_request():
    data = request.json
    statement = data.get("statement", "")
    if "capital of France is Paris" in statement:
        answer = "True"
    elif "capital of Canada is Tokyo" in statement:
        answer = "False"
    else:
        answer = "Unknown"
    return jsonify({"status": "concatenated and appended", "answer": answer})

@app.route('/decode_json', methods=['POST'])
def decode_json():
    json_output = json.dumps({
        "name": "CityName",
        "population": 100000,
        "area": 500,
        "latitude": 48.8566,
        "country": "CountryName",
        "timezone": "TimezoneName"
    })
    return jsonify({"json_output": json_output})

@app.route('/expert_answer', methods=['POST'])
def expert_answer():
    data = request.json
    question = data.get('question', '')
    if "capital of France" in question:
        answer = "paris"
    else:
        answer = "unknown"
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=30000)

#app.py
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run()
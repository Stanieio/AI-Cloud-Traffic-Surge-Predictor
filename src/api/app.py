from flask import Flask, jsonify
from src.prediction.predict_traffic import predict_traffic
from src.scaling.decision_engine import scaling_decision

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    predicted_value = predict_traffic()
    decision = scaling_decision(predicted_value)

    return jsonify({
        "predicted_traffic": float(predicted_value),
        "scaling_action": decision
    })

if __name__ == "__main__":

    print("Starting Flask server...")
    app.run(debug=True)

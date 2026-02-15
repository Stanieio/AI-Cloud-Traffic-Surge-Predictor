import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flask import Flask, jsonify
from prediction.predict_traffic import predict_traffic
from scaling.auto_scaler import decide_scaling

app = Flask(__name__)

@app.route("/predict")
def predict():
    predicted_value = predict_traffic()
    servers, action = decide_scaling(predicted_value)

    return jsonify({
        "Predicted Traffic": float(predicted_value),
        "Servers Required": servers,
        "Scaling Decision": action
    })



if __name__ == "__main__":
    app.run(debug=True)

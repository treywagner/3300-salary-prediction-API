from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model once when the app starts
model = joblib.load("salary_predict_model.ml")


@app.route("/")
def home():
    """Landing page for the Salary Prediction API"""
    return (
        "<h1>Salary Prediction API</h1>"
        "<p>BAIS:3300 - Digital Product Development</p>"
        "<p>Trey Wagner</p>"
    )


@app.route("/predict", methods=["POST"])
def predict():
    print("inside predict")

    try:
        data = request.get_json()
        print("data from user:", data)

        if data is None:
            return jsonify({"error": "No JSON received or parsing failed"}), 400

        required_fields = [
            "age", "gender", "country", "highest_deg",
            "coding_exp", "title", "company_size"
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        features = [
            int(data["age"]),
            int(data["gender"]),
            int(data["country"]),
            int(data["highest_deg"]),
            int(data["coding_exp"]),
            int(data["title"]),
            int(data["company_size"]),
        ]

        print("features before prediction:", features)
        prediction = model.predict([features])[0]
        print("prediction:", prediction)

        return jsonify({"predicted_salary": prediction})

    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({"error": str(e)}), 500


        # Ensure correct order and type
        features = [
            int(data["age"]),
            int(data["gender"]),
            int(data["country"]),
            int(data["highest_deg"]),
            int(data["coding_exp"]),
            int(data["title"]),
            int(data["company_size"]),
        ]

        print(f"features before using the model: {data}")

        prediction = model.predict([features])[0]

        print(f"prediction: {prediction}")

        return jsonify({"predicted_salary": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)

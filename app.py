
from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)

clf_model = joblib.load("model_classification.pkl")
clf_prep  = joblib.load("preprocessor_classification.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Binary encode
    df["Part_Time_Job"]   = 1 if data["Part_Time_Job"]   == "Yes" else 0
    df["Extracurricular"] = 1 if data["Extracurricular"]  == "Yes" else 0

    grade = clf_model.predict(clf_prep.transform(df))[0]
    probs = clf_model.predict_proba(clf_prep.transform(df))[0]
    classes = clf_model.classes_

    return jsonify({
        "predicted_grade": grade,
        "grade_probabilities": {c: float(p) for c, p in zip(classes, probs)}
    })

if __name__ == "__main__":
    app.run(debug=True)

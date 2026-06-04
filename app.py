from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Model
with open("playtennis_nb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Encoders
with open("playtennis_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    outlook = request.form["outlook"]
    temperature = request.form["temperature"]
    humidity = request.form["humidity"]
    wind = request.form["wind"]

    data = pd.DataFrame({
        "Outlook": [outlook],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind": [wind]
    })

    # Encode Input
    for col in data.columns:
        data[col] = encoders[col].transform(data[col])

    # Prediction
    prediction = model.predict(data)

    result = encoders["PlayTennis"].inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction_text=f"Play Tennis Prediction: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)
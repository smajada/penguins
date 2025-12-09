import pickle

from flask import Flask, jsonify, request

penguin_species = ['Adelie', 'Chinstrap', 'Gentoo']

def predict_single(model, input_data):
    prediction = model.predict([input_data])
    species = penguin_species[prediction[0]]
    return species

def predict(model, input_data):
    predictions = model.predict(input_data)
    species = [penguin_species[pred] for pred in predictions]
    return species

app = Flask("PenguinSpeciesPredictor")

@app.route("/predict_lr", methods=["POST"])
def predict_lr():
    with open("models/lr.pkl", "rb") as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    with open("models/svm.pkl", "rb") as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

@app.route("/predict_dt", methods=["POST"])
def predict_dt():
    with open("models/dt.pkl", "rb") as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

@app.route("/predict_knn", methods=["POST"])
def predict_knn():
    with open("models/knn.pkl", "rb") as f:
        dv, model = pickle.load(f)
    return predict(dv, model)

if __name__ == "__main__":
    app.run(debug=True, port = 8000)
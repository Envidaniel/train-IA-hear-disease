from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle
model = joblib.load("model_heart_disease.pkl")

# Dictionnaire pour interpréter le résultat
dicotarget = {
    1: "Le patient n'est pas malade",
    2: "Le patient est malade !"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # Stocker le résultat

    if request.method == "POST":
        try:
            # Récupérer les données du formulaire
            mv = float(request.form["mv"])
            eia = float(request.form["eia"])
            thal = float(request.form["thal"])
            oldpeak = float(request.form["oldpeak"])
            sc = float(request.form["sc"])
            cpt = float(request.form["cpt"])

            # Transformer en tableau pour la prédiction
            input_data = np.array([mv, eia, thal, oldpeak, sc, cpt]).reshape(1, -1)

            # Faire la prédiction
            prediction = dicotarget[model.predict(input_data)[0]]

        except Exception as e:
            prediction = f"Erreur : {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, request, jsonify
from joblib import load
from models.prediccion import predecir_paciente


model_rl = load("models/modelo_regresion_logistica_1.pkl")
scaler = load(r"models\scaler.pkl")

app = Flask(__name__)

@app.route("/api/hepatitis", methods=["POST"])
def calcular_prediccion_endpoint():
    if not request.is_json:
        return jsonify({"error": "El contenido debe ser JSON"}), 400
    

    data = request.get_json()
    print("JSON recibido:", data)  


    campos_requeridos = [
        "Age", "Sex_encoded", "Estado_Civil_encoded", "Ciudad_encoded", "Steroid",
        "Antivirals", "Fatigue", "Malaise", "Anorexia", "Liver_Big", "Liver_Firm",
        "Spleen_Palpable", "Spiders", "Ascites", "Varices", "Bilirubin",
        "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"
    ]

    for campo in campos_requeridos:
        if campo not in data:
            return jsonify({"error": f"Falta el campo '{campo}' en el JSON"}), 400

    try:
        valores = [float(data[campo]) for campo in campos_requeridos]

        if any(v < 0 for v in valores):
            return jsonify({"error": "Todos los valores deben ser mayores o iguales a cero"}), 400

        try:
            resultado_modelo = predecir_paciente(model_rl, scaler, valores)
        except Exception as e:
            return jsonify({"error_interno": str(e)}), 500

        return jsonify({
            "entrada": data,
            "resultado_modelo": resultado_modelo
        }), 200

    except ValueError:
        return jsonify({"error": "Todos los valores deben ser numéricos"}), 400


@app.route("/api/hepatitis/ejemplo", methods=["GET"])
def ejemplo():
    ejemplo_data = {
        "modelo": "Modelo de Hepatitis",
        "random_state": 42,
        "max_iter": 1000,
        "metricas_train": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "metricas_test": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "n_features": 21,
        "features": [
            "Age", "Sex_encoded", "Estado_Civil_encoded", "Ciudad_encoded", "Steroid",
            "Antivirals", "Fatigue", "Malaise", "Anorexia", "Liver_Big", "Liver_Firm",
            "Spleen_Palpable", "Spiders", "Ascites", "Varices", "Bilirubin",
            "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"
        ]
    }
    return jsonify(ejemplo_data), 200


if __name__ == "__main__":
    app.run(debug=True)

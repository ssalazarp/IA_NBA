from flask import Flask, render_template, request, jsonify
import numpy as np
import requests
import pandas as pd
import logging
import joblib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize the Flask app
app = Flask(__name__)

ruta_modelo = 'modelo_entrenado.pkl'
ruta_columnas = 'columnas_modelo.pkl'

modelo = joblib.load(ruta_modelo)
columnas = joblib.load(ruta_columnas)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Extraer valores del formulario
    try:
        datos = {
            "edad": int(request.json.get("edad", 0)),
            "presion_arterial_alta": int(request.json.get("presion_arterial_alta", 0)),
            "colesterol_alto": int(request.json.get("colesterol_alto", 0)),
            "imc": float(request.json.get("imc", 0.0)),
            "enfermedad_cardiaca": int(request.json.get("enfermedad_cardiaca", 0)),
            "hace_ejercicio": int(request.json.get("hace_ejercicio", 0)),
            "come_fruta": int(request.json.get("come_fruta", 0)),
            "come_vegetales": int(request.json.get("come_vegetales", 0)),
            "dificultad_caminar": int(request.json.get("dificultad_caminar", 0)),
            "genero": request.json.get("genero", 0),
            "fuma": int(request.json.get("fuma", 0))
        }
        print(f"Datos recibidos: {datos}")
    except ValueError:
        return "Error: Datos inválidos en el formulario", 400
    
    # Crear un DataFrame de entrada con los datos del formulario
    entrada = pd.DataFrame([[
        datos["presion_arterial_alta"],
        datos["colesterol_alto"],
        datos["imc"],
        datos["enfermedad_cardiaca"],
        datos["hace_ejercicio"],
        datos["come_fruta"],
        datos["come_vegetales"],
        datos["edad"],
        datos["dificultad_caminar"],
        datos["genero"],
        datos["fuma"]
    ]], columns=columnas)

    # Hacer la predicción con el modelo ML
    resultado = modelo.predict(entrada)[0]
    
    # Asignar etiquetas según el valor de la predicción
    if resultado == 0.0:
        prediccion = "No Diabético"
    elif resultado == 1.0:
        prediccion = "Pre-Diabético"
    else:
        prediccion = "Diabético"

    # Generar recomendación con Deepseek
    prompt = f"""
    Un paciente de {datos['edad']} años, con:
    - Presión arterial: {datos['presion_arterial_alta']}
    - Colesterol: {datos['colesterol_alto']}
    - IMC: {datos['imc']}
    - Enfermedad cardíaca: {datos['enfermedad_cardiaca']}
    - Hace ejercicio: {datos['hace_ejercicio']}
    - Come fruta: {datos['come_fruta']}
    - Come vegetales: {datos['come_vegetales']}
    - Tiene dificultad para caminar: {datos['dificultad_caminar']}
    - Género: {datos['genero']}
    - Fuma: {datos['fuma']}
    
    Diagnóstico: {prediccion}.
    
    Basado en estos datos, proporciona una recomendación de salud para mejorar su bienestar y prevenir complicaciones relacionadas con la diabetes.
    """
    DEEPSEEK_API_KEY = "sk-018052597ad54234bba55e93df345bcf"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un experto en salud y bienestar."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data, timeout=10)
        response.raise_for_status()
        respuesta_json = response.json()
        recomendacion = respuesta_json.get("choices", [{}])[0].get("message", {}).get("content", "No se pudo obtener la recomendación.")

    except requests.exceptions.RequestException as e:
        recomendacion = f"Error en la API: {str(e)}"

    except (KeyError, IndexError):
        recomendacion = "No se pudo obtener la recomendación en este momento."

    return jsonify({"prediccion": prediccion, "recomendacion": recomendacion})

if __name__ == "__main__":
    app.run(debug=True)


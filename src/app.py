from flask import Flask, jsonify, request
from google.oauth2 import service_account
from google.cloud import storage
import pickle

app = Flask(__name__)

@app.route('/consulta', methods=['GET'])
def consulta():
    # Obtener los datos de la solicitud
    datos = request.get_json()

    # Cargar las credenciales desde el archivo JSON
    credentials_path = './secrets/predictordiabetes-eac5e9c634e7.json'  # Reemplaza con la ruta a tu archivo de credenciales JSON
    creds = service_account.Credentials.from_service_account_file(credentials_path)

    # Conectarse a Google Cloud Storage utilizando las credenciales
    storage_client = storage.Client(credentials=creds)

    # Obtener el objeto del archivo .pkl desde Google Cloud Storage
    bucket_name = 'predictor_diabetes'  # Reemplaza con el nombre de tu bucket en Google Cloud Storage
    file_name = 'modelo_diabetes.pkl'  # Reemplaza con el nombre de tu archivo .pkl
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Descargar el archivo .pkl como bytes
    pkl_data = blob.download_as_bytes()

    # Cargar el archivo .pkl
    modelo = pickle.loads(pkl_data)

    # Realizar la consulta al modelo
    resultado = modelo.predict(datos)

    # Devolver la respuesta como JSON
    return jsonify({'resultado': resultado.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

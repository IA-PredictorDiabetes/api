from flask import Flask, jsonify, request
from google.oauth2 import service_account
from google.cloud import storage
from os import environ
import pickle

app = Flask(__name__)

print('Cargando modelo...')
# Cargar las credenciales desde el archivo JSON
# credentials_path = './secrets/predictordiabetes-eac5e9c634e7.json'  # Reemplaza con la ruta a tu archivo de credenciales JSON
# credentials_path = environ.get('GOOGLE_APPLICATION_CREDENTIALS')
credentials_path='/home/rolito/api/src/credentials.json'
creds = service_account.Credentials.from_service_account_file(credentials_path)

# Conectarse a Google Cloud Storage utilizando las credenciales
storage_client = storage.Client(credentials=creds)

# Obtener el objeto del archivo .pkl desde Google Cloud Storage
bucket_name = 'predictor_diabetes'  # Reemplaza con el nombre de tu bucket en Google Cloud Storage
file_name = 'modelo_diabetes.pkl'  # Reemplaza con el nombre de tu archivo .pkl
bucket = storage_client.get_bucket(bucket_name)
# blob = bucket.blob(file_name)

# # Descargar el archivo .pkl como bytes
# pkl_data = blob.download_as_bytes()

# Cargar el archivo .pkl
modelo = pickle.loads(bucket.blob(file_name).download_as_bytes())
print('Modelo cargado')

@app.route('/consulta', methods=['POST'])
def consulta():
    # Obtener los datos de la solicitud
    datos = request.get_json()

    # Realizar la consulta al modelo
    print('datos',datos)
    resultado = modelo.predict(datos)
    print(resultado, 'resultado')

    # Devolver la respuesta como JSON
    return jsonify({'resultado': resultado.tolist()})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'mensaje': '¡Hola, mundo!'})

if __name__ == '__main__':
    app.run(debug=True)

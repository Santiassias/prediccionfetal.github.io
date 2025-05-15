import os
import uuid
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from werkzeug.utils import secure_filename
import skfuzzy as fuzz
from scipy.stats import mode

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar dataset
df = pd.read_excel('FGR_dataset.xlsx')
X = df[[f'C{i}' for i in range(1, 31)]].values
y = df['C31'].values

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Media de C16 a C30 para predicción individual
media_c16_c30 = df[[f'C{i}' for i in range(16, 31)]].mean().values

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Rutas modelos
MODELS = {
    'log': 'modelo_logistic.pkl',
    'svm': 'modelo_svm.pkl',
    'nn': 'modelo_nn.pkl',
    'fcm': 'modelo_fcm.pkl'
}

# Entrenar y guardar modelos si no existen
if not all(os.path.exists(path) for path in MODELS.values()):
    # Regresión logística
    model_log = LogisticRegression(max_iter=2000)
    model_log.fit(X_train, y_train)
    pickle.dump(model_log, open(MODELS['log'], 'wb'))

    # SVM
    model_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model_svm.fit(X_train, y_train)
    pickle.dump(model_svm, open(MODELS['svm'], 'wb'))

    # Red neuronal
    model_nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    model_nn.fit(X_train, y_train)
    pickle.dump(model_nn, open(MODELS['nn'], 'wb'))

    # FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_train.T, c=2, m=1.5, error=0.001, maxiter=2000, seed=42
    )
    labels_fcm = np.argmax(u, axis=0)
    cluster_labels = np.zeros(2)
    for i in range(2):
        cluster_labels[i] = mode(y_train[labels_fcm == i])[0]
    pickle.dump((cntr, cluster_labels), open(MODELS['fcm'], 'wb'))

# Cargar modelo
def cargar_modelo(nombre):
    return pickle.load(open(MODELS[nombre], 'rb'))

# Guardar gráfico
def guardar_grafico(preds, reales, tipo):
    nombre = f"{tipo}_{uuid.uuid4().hex}.png"
    ruta = os.path.join(STATIC_FOLDER, nombre)
    plt.figure(figsize=(6, 4))

    if tipo == 'linea':
        plt.plot(preds, label='Predichos', color='red', marker='x')
        plt.plot(reales, label='Reales', color='blue', marker='o')
        plt.title('Predicción vs Real')
        plt.legend()
    elif tipo == 'dispersion':
        plt.scatter(reales, preds, c='green', alpha=0.5)
        plt.xlabel("Reales")
        plt.ylabel("Predichos")
        plt.title("Dispersión")
    elif tipo == 'matriz':
        cm = confusion_matrix(reales, preds)
        plt.imshow(cm, cmap='Blues')
        plt.title("Matriz de Confusión")
        plt.colorbar()
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center')

    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()
    return nombre

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    matriz = None
    exactitud = None
    grafico_linea = grafico_dispersion = grafico_matriz = None
    tipo = None  # Para controlar qué sección mostrar en el HTML

    campos = [
        'Peso materno', 'Edad gestacional', 'Presión sistólica', 'Presión diastólica',
        'Altura uterina', 'Hemoglobina', 'Latidos fetales', 'Actividad fetal',
        'Movimientos respiratorios', 'Líquido amniótico', 'Placenta',
        'Doppler arteria umbilical', 'Doppler cerebral media',
        'Doppler uterinas', 'Perfil biofísico'
    ]

    if request.method == 'POST':
        modelo_nombre = request.form['modelo']
        tipo = request.form['tipo_prediccion']

        if tipo == 'individual':
            try:
                datos = [float(request.form[f'C{i+1}']) for i in range(15)]
                datos.extend(media_c16_c30.tolist())
                datos_np = scaler.transform([datos])
                modelo = cargar_modelo(modelo_nombre)

                if modelo_nombre == 'fcm':
                    cntr, cluster_labels = modelo
                    pred_fcm = fuzz.cluster.cmeans_predict(datos_np.T, cntr, 2, error=0.005, maxiter=1000)[0]
                    cluster_idx = int(np.argmax(pred_fcm))
                    pred = int(cluster_labels[cluster_idx])
                else:
                    pred = modelo.predict(datos_np)[0]

                resultado = f"{'Sano' if pred == 0 else 'Enfermo'}"

            except Exception as e:
                resultado = f"Error en los datos: {e}"

        elif tipo == 'lote':
            archivo = request.files['archivo']
            if archivo:
                filename = secure_filename(archivo.filename)
                ruta = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                archivo.save(ruta)
                df_lote = pd.read_excel(ruta)

                columnas_necesarias = [f'C{i}' for i in range(1, 31)]
                if not all(col in df_lote.columns for col in columnas_necesarias + ['C31']):
                    resultado = "El archivo debe contener las columnas C1 a C30 y C31"
                else:
                    X_lote = df_lote[columnas_necesarias].values
                    y_lote = df_lote['C31'].values
                    X_lote_scaled = scaler.transform(X_lote)
                    modelo = cargar_modelo(modelo_nombre)

                    if modelo_nombre == 'fcm':
                        cntr, cluster_labels = modelo
                        predicciones = [
                            int(cluster_labels[np.argmax(fuzz.cluster.cmeans_predict(np.array([fila]).T, cntr, 2, error=0.005, maxiter=1000)[0])])
                            for fila in X_lote_scaled
                        ]
                    else:
                        predicciones = modelo.predict(X_lote_scaled)

                    matriz = confusion_matrix(y_lote, predicciones).tolist()
                    exactitud = round(accuracy_score(y_lote, predicciones) * 100)
                    resultado = "Predicción por lote realizada"
                    grafico_linea = guardar_grafico(predicciones, y_lote, 'linea')
                    grafico_dispersion = guardar_grafico(predicciones, y_lote, 'dispersion')
                    grafico_matriz = guardar_grafico(predicciones, y_lote, 'matriz')

    return render_template("index.html", 
                           resultado=resultado,
                           matriz=matriz,
                           exactitud=exactitud,
                           grafico_linea=grafico_linea,
                           grafico_dispersion=grafico_dispersion,
                           grafico_matriz=grafico_matriz,
                           nombres=campos,
                           tipo_prediccion=tipo)  # ESTA ES LA CLAVE

if __name__ == '__main__':
    app.run(debug=True)

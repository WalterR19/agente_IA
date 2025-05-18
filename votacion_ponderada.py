import os
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import joblib
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import scrolledtext, messagebox
import matplotlib.pyplot as plt

# Rutas de los modelos y vectorizadores
ruta_base = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio", "model_trainer")
modelo_dnn_path = os.path.join(ruta_base, "modelo_entrenado.keras")
vectorizador_dnn_path = os.path.join(ruta_base, "tfidf_vectorizer.pkl")

ruta_base = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio", "Entrenamiento_NaiveBayes")
modelo_nb_path = os.path.join(ruta_base, "modelo_naive_bayes.pkl")
vectorizador_nb_path = os.path.join(ruta_base, "vectorizador_tfidf.pkl")

# Cargar modelos y vectorizadores
modelo_dnn = load_model(modelo_dnn_path)
vectorizador_dnn = joblib.load(vectorizador_dnn_path)

modelo_nb = joblib.load(modelo_nb_path)
vectorizador_nb = joblib.load(vectorizador_nb_path)

# Pesos para la votación ponderada
W_DNN = 0.4  # Asignar mayor peso al modelo DNN si es más preciso
W_NB = 0.5   # Asignar menor peso al modelo Naive Bayes

# Crear el explicador LIME
explainer = LimeTextExplainer(class_names=["No Phishing", "Phishing"])

# Funciones para predecir
def predict_dnn(textos):
    X = vectorizador_dnn.transform(textos).toarray()
    probabilidades = modelo_dnn.predict(X)
    probabilidad_final = np.hstack([1 - probabilidades, probabilidades])
    return probabilidad_final

def predict_nb(textos):
    X = vectorizador_nb.transform(textos).toarray()
    probabilidades = modelo_nb.predict_proba(X)
    return probabilidades

# Gráfico Comparativo
def graficar_comparacion(prob_dnn, prob_nb, prob_final):
    etiquetas = ["DNN", "Naive Bayes", "Votación Ponderada"]
    valores = [prob_dnn, prob_nb, prob_final]

    plt.barh(etiquetas, valores, color=['blue', 'green', 'red'])
    plt.xlabel("Probabilidad de Phishing")
    plt.title("Comparación de Predicciones")
    plt.grid(axis='x')
    plt.show()

# Función para mostrar predicciones y explicaciones
def mostrar_explicacion():
    texto_prueba = entrada_texto.get("1.0", tk.END).strip()

    if not texto_prueba:
        messagebox.showerror("Error", "Por favor, ingrese un texto para analizar.")
        return

    # Predicciones individuales
    prob_dnn = predict_dnn([texto_prueba])[0][1]
    prob_nb = predict_nb([texto_prueba])[0][1]

    # Votación Ponderada
    prob_final = (prob_dnn * W_DNN) + (prob_nb * W_NB)

    resultado_texto.delete("1.0", tk.END)
    resultado_texto.insert(tk.END, f"📊 Texto analizado:\n{texto_prueba}\n\n")
    resultado_texto.insert(tk.END, f"🤖 DNN: {prob_dnn:.4f} ({'Phishing' if prob_dnn > 0.5 else 'No Phishing'})\n")
    resultado_texto.insert(tk.END, f"📈 Naive Bayes: {prob_nb:.4f} ({'Phishing' if prob_nb > 0.5 else 'No Phishing'})\n")
    resultado_texto.insert(tk.END, f"⚖️ Votación Ponderada: {prob_final:.4f} ({'Phishing' if prob_final > 0.5 else 'No Phishing'})\n\n")

    # Explicación con LIME
    exp_dnn = explainer.explain_instance(texto_prueba, predict_dnn, num_features=10)
    exp_nb = explainer.explain_instance(texto_prueba, predict_nb, num_features=10)

    # Guardar explicaciones en HTML
    ruta_html_dnn = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio", "explicacion_DNN.html")
    ruta_html_nb = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio", "explicacion_NB.html")
    
    exp_dnn.save_to_file(ruta_html_dnn)
    exp_nb.save_to_file(ruta_html_nb)

    resultado_texto.insert(tk.END, f"✅ Explicación DNN guardada en: {ruta_html_dnn}\n")
    resultado_texto.insert(tk.END, f"✅ Explicación Naive Bayes guardada en: {ruta_html_nb}\n")

    # Mostrar gráfico comparativo
    graficar_comparacion(prob_dnn, prob_nb, prob_final)

    messagebox.showinfo("Explicaciones Guardadas", f"Las explicaciones se guardaron en:\n{ruta_html_dnn}\n{ruta_html_nb}")

# Interfaz gráfica en Tkinter
ventana = tk.Tk()
ventana.title("Comparación y Votación Ponderada: DNN vs Naive Bayes")
ventana.geometry("600x500")

tk.Label(ventana, text="Ingrese el texto para analizar:", font=("Arial", 12)).pack(pady=5)
entrada_texto = scrolledtext.ScrolledText(ventana, width=60, height=5)
entrada_texto.pack(pady=10)

tk.Button(ventana, text="Generar Comparación", command=mostrar_explicacion).pack(pady=5)

resultado_texto = scrolledtext.ScrolledText(ventana, width=60, height=15, wrap=tk.WORD)
resultado_texto.pack(pady=10)

ventana.mainloop()

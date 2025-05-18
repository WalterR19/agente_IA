import os
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
import joblib
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox

# Ruta de carga del modelo y vectorizador
ruta_base = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio", "model_trainer")
modelo_path = os.path.join(ruta_base, "modelo_entrenado.keras")
vectorizador_path = os.path.join(ruta_base, "tfidf_vectorizer.pkl")

# Cargar el modelo y el vectorizador
modelo = load_model(modelo_path)
vectorizador_tfidf = joblib.load(vectorizador_path)

# Crear el explicador LIME
explainer = LimeTextExplainer(class_names=["No Phishing", "Phishing"])

# Función que predice usando el modelo DNN
def predict_fn(textos):
    X = vectorizador_tfidf.transform(textos).toarray()
    probabilidades = modelo.predict(X)
    probabilidad_final = np.hstack([1 - probabilidades, probabilidades])
    return probabilidad_final

# Función para mostrar la explicación en Tkinter
def mostrar_explicacion():
    texto_prueba = entrada_texto.get("1.0", tk.END).strip()

    if not texto_prueba:
        messagebox.showerror("Error", "Por favor, ingrese un texto para analizar.")
        return

    exp = explainer.explain_instance(texto_prueba, predict_fn, num_features=10)

    # Mostrar en ventana Tkinter
    resultado_texto.delete("1.0", tk.END)
    resultado_texto.insert(tk.END, f"📊 Texto analizado:\n{texto_prueba}\n\n")
    resultado_texto.insert(tk.END, f"🟩 Palabras destacadas (positivas): {exp.as_list()}\n")

    # Guardar la explicación en un archivo HTML
    ruta_html = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio", "explicacion_LIME.html")
    exp.save_to_file(ruta_html)
    messagebox.showinfo("Explicación Guardada", f"Explicación guardada en: {ruta_html}")

# Interfaz gráfica en Tkinter
ventana = tk.Tk()
ventana.title("Explicación del Modelo con LIME")
ventana.geometry("600x500")

tk.Label(ventana, text="Ingrese el texto para analizar:", font=("Arial", 12)).pack(pady=5)
entrada_texto = scrolledtext.ScrolledText(ventana, width=60, height=5)
entrada_texto.pack(pady=10)

tk.Button(ventana, text="Generar Explicación", command=mostrar_explicacion).pack(pady=5)

resultado_texto = scrolledtext.ScrolledText(ventana, width=60, height=15, wrap=tk.WORD)
resultado_texto.pack(pady=10)

ventana.mainloop()

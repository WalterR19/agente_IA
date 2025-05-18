import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import MaxAbsScaler
from nltk.corpus import stopwords
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import BatchNormalization
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Embedding, Bidirectional
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
import joblib
import keras
from keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tkinter import ttk, messagebox
from tkinter import messagebox, ttk, scrolledtext, filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

class EntrenadorRedProfunda:
    def __init__(self, columnas_seleccionadas, columna_objetivo, tabla_seleccionada, df):
        self.columnas_seleccionadas = columnas_seleccionadas
        self.columna_objetivo = columna_objetivo
        self.df = df
        self.data_loaded = False
        self.tokenization_done = False
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()
        self.tabla_seleccionada = tabla_seleccionada
        self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # Tokenizador para la LSTM
        self.max_length = 300  # Longitud máxima de secuencias
       
               
    def limpiar_texto(self, texto):
        """Limpia el texto pero conserva información útil."""
        texto = texto.lower()
        texto = re.sub(r'[^\w\s@#_-]', '', texto)  # Mantiene caracteres especiales relevantes
        palabras = word_tokenize(texto)  # Tokenización por palabras
        texto = ' '.join([stemmer.stem(word) for word in palabras])  # Stemming de cada palabra
        return texto

    def cargar_y_preprocesar_datos(self):
        """Preprocesa los datos recibidos del agente de IA."""
        if not self.data_loaded:
            try:
                print(f"📌 Columnas seleccionadas: {self.columnas_seleccionadas}")
                print(f"🎯 Columna objetivo: {self.columna_objetivo}")

                # ✅ Usar directamente los datos que ya fueron extraídos
                if self.df is None or self.df.empty:
                    raise ValueError("❌ El DataFrame recibido está vacío. Verifique la extracción de datos en el agente.")

                # Mantener solo columnas seleccionadas y tipo `object` (texto)
                self.df = self.df[self.columnas_seleccionadas + [self.columna_objetivo]]
                self.df = self.df.loc[:, self.df.dtypes == 'object']

                # Eliminar valores NaN en la columna objetivo
                if self.columna_objetivo in self.df:
                    print("Verificando valores NaN en la columna objetivo...")
                    print(self.df[self.columna_objetivo].isna().sum(), "valores NaN en la columna objetivo.")

                    self.df = self.df.dropna(subset=[self.columna_objetivo])
                    self.df[self.columna_objetivo] = self.df[self.columna_objetivo].astype(str)  # Asegurarse de que la columna objetivo sea de tipo string

                concatenated_text = self.df[self.columnas_seleccionadas + [self.columna_objetivo]].fillna('').astype(str).apply(lambda row: ' '.join(row), axis=1)
                self.df['text'] = concatenated_text
                print("🔍 Datos concatenados para entrenamiento:")
                print(self.df["text"].head())

                # Guardar distribución original de etiquetas
                self.distribucion_original = self.df[self.columna_objetivo].value_counts()
                print("📊 Distribución original de etiquetas:")
                print(self.distribucion_original)

                # Establecer el flag de carga
                self.data_loaded = True

                if not self.tokenization_done:
                    print("Tokenización pendiente. Iniciando...")
                    self.tokenize_data()

                print("Datos cargados y preprocesados.")
            except Exception as e:
                print(f"Error al cargar y preprocesar los datos: {e}")
        else:
            print("Datos ya cargados.")

    def tokenize_data(self):
        if not self.data_loaded:
            print("Datos no cargados. No se puede tokenizar.")
            return

        if self.tokenization_done:
            print("Tokenización ya realizada.")
            return

        try:
            print("Iniciando la tokenización de datos...")

            texts = self.df['text'].tolist()

            # Verificar y mapear valores de la columna objetivo
            if self.columna_objetivo in self.df:
                le = LabelEncoder()
                self.df[self.columna_objetivo] = le.fit_transform(self.df[self.columna_objetivo])
                labels = self.df[self.columna_objetivo].tolist()
                print("Etiquetas mapeadas:", le.classes_)
            else:
                print(f"Columna {self.columna_objetivo} no encontrada en los datos.")
                return

            # Tokenización con TF-IDF
            X = self.tfidf_vectorizer.fit_transform(texts).toarray()
            y = np.array(labels)

            # Verificar si hay NaNs en las etiquetas
            if np.any(np.isnan(y)):
                print("Error: Etiquetas contienen valores NaN.")
                return

            # Aplicar SMOTE a todo el conjunto de datos antes de dividir
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print("Distribución después de SMOTE:", np.bincount(y_balanced))

            # Aumentar el tamaño del conjunto de prueba para alcanzar alrededor de 3000 ejemplos por etiqueta
            X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.6, random_state=42, stratify=y_balanced)

            # Guardar los datos de entrenamiento y prueba para su uso posterior
            np.save("X_train.npy", X_train)
            np.save("y_train.npy", y_train)
            np.save("X_test.npy", X_test)
            np.save("y_test.npy", y_test)

            self.tokenization_done = True
            print("Tokenización completada.")

            # Llamar a entrenar_modelo después de tokenizar los datos
            self.entrenar_modelo()

        except Exception as e:
            print(f"Error durante la tokenización: {e}")
    
    def entrenar_modelo(self):
        if not self.tokenization_done:
            print("No se ha realizado la tokenización. Imposible entrenar el modelo.")
            return

        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")

        # Inicializar y ajustar el LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(y_train)

        print("🔹 Iniciando entrenamiento del modelo...")

        # **Corrección del Optimizer (sin LearningRateSchedule)**
        optimizer = Adam(learning_rate=0.001)  # Se usa un valor fijo ajustable

        # **Mejoras clave en la arquitectura**
        self.model = Sequential([
            Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.5),

            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),

            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),

            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # **Callbacks mejorados**
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        # **Balanceo de clases**
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))

        # **Entrenamiento**
        history = self.model.fit(
            X_train, y_train,
            epochs=50,  
            batch_size=64,  
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],  
            class_weight=class_weights_dict
        )

        print("✅ Entrenamiento completado.")
        
        self.guardar_entrenamiento()

        # **Corrección: Ahora se pasa `history` a la función de gráficos**
        self.graficar_historial(history)

        # **Evaluar el modelo después del entrenamiento**
        self.evaluar_modelo()

        # **Mostrar los valores finales**
        final_epoch = len(history.history['loss']) - 1
        final_train_loss = history.history['loss'][final_epoch]
        final_val_loss = history.history['val_loss'][final_epoch]
        final_train_accuracy = history.history['accuracy'][final_epoch]
        final_val_accuracy = history.history['val_accuracy'][final_epoch]

        print(f"\n📌 Valores finales del entrenamiento:")
        print(f"📉 Pérdida de entrenamiento: {final_train_loss:.4f}")
        print(f"📉 Pérdida de validación: {final_val_loss:.4f}")
        print(f"✅ Precisión de entrenamiento: {final_train_accuracy:.4f}")
        print(f"✅ Precisión de validación: {final_val_accuracy:.4f}")

        print("🚀 Entrenamiento y evaluación completados.")

    def evaluar_modelo(self):
        if not self.model:
            print("Modelo no entrenado. Imposible evaluar.")
            return

        # Cargar los datos de prueba
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")

        # Realizar predicciones
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calcular mejor umbral
        mejor_umbral = self.calcular_mejor_umbral(y_test, y_pred_proba)
        print(f"📌 Mejor umbral calculado: {mejor_umbral:.4f}")

        # Evaluar el modelo
        print("\n📊 Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # Confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\n🛑 Matriz de confusión:")
        print(conf_matrix)

        # Graficar la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión')
        plt.show()

        # Curva ROC y AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid(True)  
        plt.show()

        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precisión')
        plt.title('Curva Precision-Recall')
        plt.grid(True)  
        plt.show()

        print("✅ Evaluación completada.")

    def calcular_mejor_umbral(self, y_true, y_pred_proba):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)  # Evita divisiones por 0
        mejor_umbral = thresholds[np.argmax(f1_scores)]
        print(f"📌 Mejor umbral calculado: {mejor_umbral:.4f}")
        return mejor_umbral

    def graficar_historial(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Historial de Entrenamiento')
        plt.legend()
        plt.grid(True)  
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Precisión de validación')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.title('Historial de Precisión')
        plt.legend()
        plt.grid(True)  
        plt.show()
        
#---------------------------PREDICCIONES---------------------------------------

    def guardar_entrenamiento(self):
        try:
            # 📌 Define las rutas en la función (Si no están ya en self)
            desktop_path = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio")
            model_dir = os.path.join(desktop_path, "model_trainer")
            os.makedirs(model_dir, exist_ok=True)  # Crear la carpeta si no existe

            # 📌 Define rutas específicas para cada componente
            model_path = os.path.join(model_dir, "modelo_entrenado.keras")
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
            label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

            # 📌 Guardar modelo, vectorizador y label encoder
            self.model.save(model_path)
            joblib.dump(self.tfidf_vectorizer, vectorizer_path)
            joblib.dump(self.le, label_encoder_path)

            print(f"✅ Modelo guardado en: {model_path}")
            print("✅ Vectorizador TF-IDF y Label Encoder guardados correctamente.")

        except Exception as e:
            print(f"❌ Error al guardar el modelo y los componentes: {e}")
            
   
        
        
    
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE, ADASYN
import joblib
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

nltk.download("stopwords")
nltk.download("punkt")
stemmer = SnowballStemmer("spanish")
stop_words = set(stopwords.words("spanish"))

class NaiveBayesTrainer:
    def __init__(self, columnas_seleccionadas, columna_objetivo, tabla_seleccionada, df):
        self.columnas_seleccionadas = columnas_seleccionadas
        self.columna_objetivo = columna_objetivo
        self.tabla_seleccionada = tabla_seleccionada
        self.df = df
        self.tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=3, max_df=0.9)
        self.model = MultinomialNB(alpha=0.005, fit_prior=True)  # 🔥 Alpha más bajo para mejorar precisión

    def limpiar_texto(self, texto):
        """Limpia el texto pero conserva URLs, correos y menciones."""
        texto = texto.lower()
        texto = re.sub(r'[^\w\s@#_.-]', '', texto)  # NO elimina URLs ni correos
        palabras = nltk.word_tokenize(texto)
        texto = ' '.join([stemmer.stem(word) for word in palabras if word not in stop_words])
        return texto

    def cargar_y_preprocesar_datos_nb(self):
        try:
            print("📌 Columnas seleccionadas:", self.columnas_seleccionadas)
            print("🎯 Columna objetivo:", self.columna_objetivo)

            if self.df is None or self.df.empty:
                raise ValueError("❌ El DataFrame recibido está vacío.")

            self.df = self.df[self.columnas_seleccionadas + [self.columna_objetivo]].astype(str).fillna('')
            self.df['text'] = self.df.apply(lambda row: ' '.join(row), axis=1)

            le = LabelEncoder()
            self.df[self.columna_objetivo] = le.fit_transform(self.df[self.columna_objetivo])

            X = self.tfidf_vectorizer.fit_transform(self.df['text']).toarray()
            y = self.df[self.columna_objetivo]

            # Aplicar ADASYN en lugar de SMOTE si el desbalance es muy alto
            if np.bincount(y)[0] / np.bincount(y)[1] > 2:
                print("⚠️ Desbalance severo detectado. Usando ADASYN en lugar de SMOTE.")
                balanceador = ADASYN(random_state=42)
            else:
                balanceador = SMOTE(random_state=42, k_neighbors=3)

            X_balanced, y_balanced = balanceador.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

            np.save("X_nb_train.npy", X_train)
            np.save("y_nb_train.npy", y_train)
            np.save("X_nb_test.npy", X_test)
            np.save("y_nb_test.npy", y_test)

            print("✅ Datos preprocesados y guardados correctamente.")
            self.crear_carpeta_entrenamiento()
            self.entrenar_naive_bayes()
            
        except Exception as e:
            print(f"❌ Error durante la carga y preprocesamiento de datos: {e}")

    def entrenar_naive_bayes(self):
        try:
            print("🚀 Iniciando entrenamiento de Naive Bayes...")
            X_train = np.load("X_nb_train.npy")
            y_train = np.load("y_nb_train.npy")

            self.model.fit(X_train, y_train)

            scores = cross_val_score(self.model, X_train, y_train, cv=5)
            print(f"🚀 Precisión media en validación cruzada: {scores.mean():.2f}")

            joblib.dump(self.model, "modelo_naive_bayes.pkl")
            joblib.dump(self.tfidf_vectorizer, "vectorizador_tfidf.pkl")

            print("✅ Modelo Naive Bayes entrenado y guardado correctamente.")
            self.evaluar_modelo()
        except Exception as e:
            print(f"❌ Error durante el entrenamiento de Naive Bayes: {e}")

    def evaluar_modelo(self):
        try:
            print("📊 Evaluando el modelo Naive Bayes...")

            X_test = np.load("X_nb_test.npy")
            y_test = np.load("y_nb_test.npy")

            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            optimal_threshold = thresholds[np.argmax(2 * (precision * recall) / (precision + recall))]

            print(f"📌 Mejor umbral calculado: {optimal_threshold:.2f}")
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)

            print("\n📊 Reporte de clasificación:")
            print(classification_report(y_test, y_pred, zero_division=1))

            conf_matrix = confusion_matrix(y_test, y_pred)
            print("\n🛑 Matriz de confusión:")
            print(conf_matrix)

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.title('Matriz de Confusión - Naive Bayes')
            plt.show()

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - Naive Bayes')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

            print("✅ Evaluación del modelo completada.")
        except Exception as e:
            print(f"❌ Error durante la evaluación del modelo Naive Bayes: {e}")

    def crear_carpeta_entrenamiento(self):
        ruta_escritorio = os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio")
        self.ruta_entrenamiento = os.path.join(ruta_escritorio, "Entrenamiento_NaiveBayes")
        os.makedirs(self.ruta_entrenamiento, exist_ok=True)
        print(f"📂 Carpeta de entrenamiento creada en: {self.ruta_entrenamiento}")

    def guardar_archivos(self):
        np.save(os.path.join(self.ruta_entrenamiento, "X_nb_train.npy"), np.load("X_nb_train.npy"))
        np.save(os.path.join(self.ruta_entrenamiento, "y_nb_train.npy"), np.load("y_nb_train.npy"))
        np.save(os.path.join(self.ruta_entrenamiento, "X_nb_test.npy"), np.load("X_nb_test.npy"))
        np.save(os.path.join(self.ruta_entrenamiento, "y_nb_test.npy"), np.load("y_nb_test.npy"))
        joblib.dump(self.model, os.path.join(self.ruta_entrenamiento, "modelo_naive_bayes.pkl"))
        joblib.dump(self.tfidf_vectorizer, os.path.join(self.ruta_entrenamiento, "vectorizador_tfidf.pkl"))
        print("✅ Entrenamiento guardado correctamente en la carpeta correspondiente.")

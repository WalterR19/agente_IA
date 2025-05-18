import tkinter as tk
import mysql.connector
import numpy as np
import pandas as pd
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from langdetect import DetectorFactory
from tkinter import messagebox
import os
from red_neuronal_profunda import EntrenadorRedProfunda  
from naive_bayes import NaiveBayesTrainer
#from predictor import Predictor

nltk.download('stopwords')
nltk.download('punkt')
DetectorFactory.seed = 0

class EmailClassifierAgent:
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_connection = None
        self.tokenizer = Tokenizer(num_words=5000)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.stop_words = set(stopwords.words('spanish')).union(set(stopwords.words('english')))
        self.stemmer = SnowballStemmer('spanish')
        self.tabla_seleccionada = None
        self.columna_objetivo = None
        self.columnas_seleccionadas = []
        
        #self.predictor = Predictor() 
         
        self.create_database_connection()
        self.mostrar_seleccion_tabla()
        
        

    def create_database_connection(self):
        try:
            self.db_connection = mysql.connector.connect(
                host="localhost",
                port=3306,
                user="root",
                password="Walter1993##",
                database="phishing",
                ssl_disabled=True
            )
            if self.db_connection.is_connected():
                print("Conexión a la base de datos establecida correctamente.")
                messagebox.showinfo("Conexión", "Conexión a la base de datos establecida correctamente.")
        except mysql.connector.Error as e:
            print(f"Error al conectar con MySQL: {e}")
            messagebox.showerror("Error de Conexión", f"Error al conectar con MySQL: {e}")

    def mostrar_seleccion_tabla(self):
        self.tablas = self.obtener_tablas()

        if not self.tablas:
            print("No se encontraron tablas en la base de datos.")
            return

        self.ventana_tablas = tk.Tk()
        self.ventana_tablas.title("Seleccionar Tabla para Entrenar Red Neuronal")

        self.listbox_tablas = tk.Listbox(self.ventana_tablas, selectmode=tk.SINGLE)
        for tabla in self.tablas:
            self.listbox_tablas.insert(tk.END, tabla)
        self.listbox_tablas.pack()

        tk.Button(self.ventana_tablas, text="Seleccionar", command=self.seleccionar_tabla).pack(pady=10)
        tk.Button(self.ventana_tablas, text="Finalizar", command=self.ventana_tablas.destroy).pack(pady=10)
        
    def obtener_tablas(self):
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SHOW TABLES")
            tablas = [tabla[0] for tabla in cursor.fetchall()]
            cursor.close()
            return tablas
        except mysql.connector.Error as e:
            print(f"Error al obtener las tablas: {e}")
            messagebox.showerror("Error", f"Error al obtener las tablas: {e}")
            return []

    def seleccionar_tabla(self):
        seleccion = self.listbox_tablas.curselection()
        if not seleccion:
            print("No se ha seleccionado ninguna tabla.")
            return

        self.tabla_seleccionada = self.tablas[seleccion[0]]
        self.ventana_tablas.destroy()
        self.mostrar_seleccion_columnas()

    def mostrar_seleccion_columnas(self):
        if not self.tabla_seleccionada:
            print("Primero debe seleccionar una tabla.")
            return

        self.ventana_columnas = tk.Toplevel()
        self.ventana_columnas.title("Seleccionar Columnas de Entrada")

        cursor = self.db_connection.cursor()
        try:
            cursor.execute(f"SHOW COLUMNS FROM {self.tabla_seleccionada}")
            columnas = [i[0] for i in cursor.fetchall() if i[0].lower() != 'id']

            tk.Label(self.ventana_columnas, text="Seleccione las columnas de entrada:").pack(pady=10)

            self.listbox_columnas = tk.Listbox(self.ventana_columnas, selectmode=tk.MULTIPLE)
            for col in columnas:
                self.listbox_columnas.insert(tk.END, col)
            self.listbox_columnas.pack()

            tk.Button(self.ventana_columnas, text="Siguiente", command=self.seleccionar_columnas).pack(pady=10)

        except mysql.connector.Error as e:
            print(f"Error al obtener columnas: {e}")
        finally:
            cursor.close()
        
    def seleccionar_columnas(self):
        seleccion_columnas = self.listbox_columnas.curselection()

        if not seleccion_columnas:
            print("Debe seleccionar al menos una columna de entrada.")
            return
        elif len(seleccion_columnas) > 5:
            print("Puede seleccionar un máximo de cinco columnas de entrada.")
            return

        self.columnas_seleccionadas = [self.listbox_columnas.get(i) for i in seleccion_columnas]
        self.ventana_columnas.destroy()

        self.mostrar_seleccion_columna_objetivo()

    def mostrar_seleccion_columna_objetivo(self):
        if not self.tabla_seleccionada:
            print("Primero debe seleccionar una tabla.")
            return

        self.ventana_columna_objetivo = tk.Toplevel()
        self.ventana_columna_objetivo.title("Seleccionar Columna Objetivo")

        cursor = self.db_connection.cursor()
        try:
            cursor.execute(f"SHOW COLUMNS FROM {self.tabla_seleccionada}")
            columnas = [i[0] for i in cursor.fetchall() if i[0].lower() != 'id']

            tk.Label(self.ventana_columna_objetivo, text="Seleccione la columna objetivo:").pack(pady=10)

            self.listbox_objetivo = tk.Listbox(self.ventana_columna_objetivo, selectmode=tk.SINGLE)
            for col in columnas:
                self.listbox_objetivo.insert(tk.END, col)
            self.listbox_objetivo.pack()

            tk.Button(self.ventana_columna_objetivo, text="Siguiente", command=self.guardar_seleccion_columnas).pack(pady=10)

        except mysql.connector.Error as e:
            print(f"Error al obtener columnas: {e}")
        finally:
            cursor.close()

    def guardar_seleccion_columnas(self):
        # Obtener la columna objetivo seleccionada desde el Listbox
        seleccion_objetivo = self.listbox_objetivo.curselection()
        if seleccion_objetivo:
            self.columna_objetivo = self.listbox_objetivo.get(seleccion_objetivo[0])
        else:
            self.columna_objetivo = None

        if not self.columna_objetivo:
            print("Debe seleccionar una columna objetivo.")
            return

        if not self.columnas_seleccionadas:
            print("Debe seleccionar al menos una columna de entrada.")
            return

     
        self.extraer_y_entrenar_modelo()

    def iniciar_predictor(self):
        """Iniciar predictor de forma independiente"""
        try:
            predictor = Predictor(
                self.columnas_seleccionadas,
                self.columna_objetivo,
                self.tabla_seleccionada,
                self.df
            )
            predictor.cargar_modelo()
            predictor.abrir_ventana_prediccion()  # Abre la interfaz gráfica del predictor
        except Exception as e:
            print(f"❌ Error al iniciar el predictor: {e}")


    def extraer_y_entrenar_modelo(self):
   
        try:
            cursor = self.db_connection.cursor()
            columnas_sql = ', '.join([f"`{col}`" for col in self.columnas_seleccionadas + [self.columna_objetivo]])
            query = f"SELECT {columnas_sql} FROM `{self.tabla_seleccionada}`"

            cursor.execute(query)
            datos = cursor.fetchall()

            df = pd.DataFrame(datos, columns=self.columnas_seleccionadas + [self.columna_objetivo])

            print("✅ Datos extraídos correctamente. Iniciando red neuronal profunda...")

            # ✅ ENTRENAR LA RED NEURONAL PROFUNDA
            try:
                entrenador = EntrenadorRedProfunda(self.columnas_seleccionadas, self.columna_objetivo, self.tabla_seleccionada, df)
                entrenador.cargar_y_preprocesar_datos()
            except Exception as e:
                print(f"❌ Error al ejecutar la red neuronal profunda: {e}")
                return  # Detener el proceso si hay un error

            print("✅ Red Neuronal Profunda entrenada correctamente.")
            
            # ✅ INICIAR AUTOMÁTICAMENTE LA RED RECURRENTE
            print("🚀 Iniciando entrenamiento del modelo Naive Bayes...")
            try:
                nb_trainer = NaiveBayesTrainer(self.columnas_seleccionadas, self.columna_objetivo, self.tabla_seleccionada, df)
                nb_trainer.cargar_y_preprocesar_datos_nb()
            except Exception as e:
                print(f"❌ Error al ejecutar el modelo Naive Bayes: {e}")
            print("✅ Modelo Naive Bayes entrenado correctamente.")
            #print("🚨 Iniciando el Predictor...")

            #try:
                #predictor = Predictor(self.columnas_seleccionadas, self.columna_objetivo, self.tabla_seleccionada, df)
                #predictor.cargar_modelo()
                #predictor.abrir_ventana_prediccion()  # Abre la interfaz gráfica del predictor
            #except Exception as e:
                #print(f"❌ Error al iniciar el predictor: {e}")
                #return

            #print("✅ Predictor iniciado correctamente. Puedes realizar predicciones.")


        except mysql.connector.Error as e:
            print(f"❌ Error al extraer los datos de MySQL: {e}")

        except Exception as e:
            print(f"❌ Error inesperado: {e}")


# Iniciar el agente
db_config = {}
agent = EmailClassifierAgent(db_config)
tk.mainloop()
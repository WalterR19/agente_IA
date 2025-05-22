# 🛡️ Sistema Híbrido de Detección de Phishing con DNN y Naive Bayes

Este proyecto implementa un sistema de clasificación de correos electrónicos para identificar ataques de phishing, integrando dos enfoques de Machine Learning complementarios: una Red Neuronal Profunda (DNN) y un modelo estadístico Naive Bayes, coordinados por un **Agente de Clasificación** que gestiona el entrenamiento y la base de datos. La predicción se realiza a través de un **algoritmo de votación ponderada** y se presenta en una interfaz gráfica amigable.

---



### 🧩 Componentes principales:

---

### 👨‍💻 Usuario
Interactúa con el sistema a través de una GUI donde puede ingresar texto (ej. correos sospechosos) para análisis en tiempo real.

---

### 🖥️ GUI (Interfaz Gráfica)
- `interfazUsuario()`: Presenta la ventana al usuario.
- `enviarDatos()`: Envía el texto ingresado a los modelos para análisis.
- Se comunica con los módulos de predicción, votación y explicabilidad.

---

### 🧠 VotacionPonderada
- Atributos:
  - `modeloDNN`, `modeloNB`: Instancias de los modelos entrenados.
  - `pesoDNN`, `pesoNB`: Pesos configurables para la votación.
- Método:
  - `combinarPredicciones()`: Combina los resultados de ambos modelos para una decisión final.
- ⚖️ **Funciona como el núcleo decisor del sistema.**

---

### 🔎 ExplicadorLIME
- `generarExplicacion(texto)`: Genera una explicación interpretativa de por qué el modelo clasificó el texto como phishing o no.
- 📊 Esto mejora la **transparencia del sistema** ante el usuario.

---

### 🤖 PredictDNN y PredictNB
Ambos reciben el texto procesado por la GUI:

- `predecir(texto)`:
  - En `PredictDNN`: Utiliza la red neuronal entrenada con TensorFlow/Keras.
  - En `PredictNB`: Usa un clasificador Naive Bayes entrenado con Scikit-learn.
- Devuelven probabilidades de phishing, que luego se combinan.

---

### 🧠 ModelosEntrenados
- `modeloDNN`, `modeloNB`: Contienen los modelos previamente entrenados y cargados en memoria.
- Se usan para no reentrenar en cada ejecución.

---

### 🧮 RedNeuronal y NaiveBayes
- Ambas clases poseen:
  - `entrenar()`: Usa datos desde MySQL para generar un modelo.
  - `guardar()`: Persiste el modelo entrenado en disco.
- Diferencia:
  - `RedNeuronal`: Captura relaciones profundas con capas ocultas.
  - `NaiveBayes`: Es rápido y efectivo en textos simples.

---

### 🧠 AgenteClasificacion
- Controlador central de entrenamiento.
- Atributos:
  - `dbConfig`, `dbConnection`: Configura conexión a base de datos.
- Método:
  - `entrenarModelos()`: Extrae datos desde `MySQL`, entrena ambos modelos y guarda sus versiones finales.

---

### 💾 MySQL
- `guardarDatos()`: Permite registrar textos nuevos y sus etiquetas.
- `obtenerDatos()`: Recupera correos y clasificaciones históricas para reentrenamiento.

---

## 🧪 Pruebas Implementadas

- Pruebas con `pytest` verifican:
  - Robustez del algoritmo de votación frente a texto vacío, numérico, con símbolos y frases reales.
  - Precisión del modelo ante diferentes escenarios de entrada.
  - Integridad del modelo híbrido completo.

---

## 🚀 Ejecución

```bash
python votacion_ponderada.py

# Benchmark de Modelos de Lenguaje Open-Source para RAG Local sobre PDFs

Este proyecto realiza una evaluación comparativa de modelos de lenguaje open-source usando un sistema de RAG (Retrieval-Augmented Generation) local para responder preguntas sobre documentos PDF en español.

## 🔧 Requisitos

### Python

* Python 3.10 o superior

### Dependencias

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### Modelos en Ollama

Asegúrate de tener [Ollama](https://ollama.com) instalado y funcionando. Luego, descarga los modelos:

```bash
ollama pull llama3
ollama pull mistral
ollama pull zephyr
ollama pull phi3
ollama pull openhermes
ollama pull nomic-embed-text  # Para embeddings
```

## 📂 Estructura del Proyecto

```
.
├── benchmark_rag_avanzado.py
├── rag_core.py
├── requirements.txt
├── README.md
├── preguntas.txt
├── docs/               # PDFs para evaluar
│   ├── tesis_05p.pdf
│   ├── tesis_10p.pdf
│   └── ...
├── vectorstore/        # Se genera automáticamente
├── resultados_benchmark.csv
├── resultados_benchmark.xlsx
```

## 🚀 Ejecución

1. Coloca los archivos PDF en la carpeta `docs/`
2. Asegúrate de tener el archivo `preguntas.txt` con preguntas base
3. Ejecuta el benchmark:

```bash
python benchmark_rag_avanzado.py
```

4. Espera a que finalice. Se generarán dos archivos de resultados:

   * `resultados_benchmark.csv`
   * `resultados_benchmark.xlsx`

## 📊 Métricas recolectadas por modelo, PDF y pregunta

* Latencia de respuesta
* Uso de RAM y CPU (antes y después)
* Coherencia estructural de la respuesta
* Respuesta generada completa

## 📄 Evaluación

Puedes abrir el Excel generado y agregar manualmente columnas para:

* Precisión (0, 0.5, 1)
* Claridad
* Observaciones

Esto permitirá generar gráficos y tablas comparativas para el artículo científico.

## 📊 Aplicaciones

Este benchmark sirve como base para seleccionar el mejor modelo open-source para:

* Aplicaciones tipo ChatPDF pero offline
* Herramientas seguras para documentos confidenciales
* Sistemas de QA automáticos en educación, investigación y gobierno

---

📅 Proyecto desarrollado para evaluación experimental en el contexto del artículo a presentarse en COINCITEC 2025.

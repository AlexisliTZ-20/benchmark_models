import os
import time
import psutil
import pandas as pd
from rag_core import process_pdf, get_qa_chain

# Intenta cargar soporte de GPU (opcional)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_DISPONIBLE = True
except Exception:
    GPU_DISPONIBLE = False

# Configuración de modelos instalados
MODELOS = {
    "llama3": "llama3",
    "mistral": "mistral",
    "zephyr": "zephyr",
    "phi3": "phi3",
    "openhermes": "openhermes"
}

# Cargar preguntas desde un solo archivo
with open("preguntas.txt", "r", encoding="utf-8") as f:
    PREGUNTAS = [line.strip() for line in f if line.strip()]

# Crear lista de resultados
resultados = []

# Recorrer todos los PDFs en la carpeta docs/
for pdf_file in os.listdir("docs"):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join("docs", pdf_file)
    pdf_id = os.path.splitext(pdf_file)[0]  # nombre sin extensión

    print(f"\nProcesando PDF: {pdf_file}")

    # Solo procesar si no existe el vectorstore
    if not os.path.exists(f"vectorstore/{pdf_id}"):
        print("Indexando PDF...")
        process_pdf(pdf_path, pdf_id)

    for nombre_modelo, modelo in MODELOS.items():
        print(f"\nEvaluando modelo: {nombre_modelo} con {pdf_file}")
        qa_chain = get_qa_chain(pdf_id, modelo)

        for pregunta in PREGUNTAS:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024**2
            cpu_before = process.cpu_percent(interval=0.1)

            if GPU_DISPONIBLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_before = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            else:
                gpu_before = None

            prompt = (
                "Responde en español. Primero, razona paso a paso, mostrando tu pensamiento antes de responder. "
                "Al final, da la respuesta clara comenzando con: RESPUESTA FINAL: "
                f"\n\nPregunta: {pregunta}"
            )

            start = time.perf_counter()
            respuesta = qa_chain.run(prompt)
            latencia = time.perf_counter() - start

            mem_after = process.memory_info().rss / 1024**2
            cpu_after = process.cpu_percent(interval=0.1)

            if GPU_DISPONIBLE:
                gpu_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            else:
                gpu_after = None

            respuesta_lower = respuesta.lower()
            coherencia = 1 if "respuesta final" in respuesta_lower else 0.5 if respuesta else 0
            longitud_respuesta = len(respuesta)
            contiene_error = int(any(x in respuesta_lower for x in [
                "no lo sé", "no se encuentra", "no aparece", "no puedo", "no está disponible"
            ]))

            resultados.append({
                "modelo": nombre_modelo,
                "pdf": pdf_file,
                "pregunta": pregunta,
                "respuesta": respuesta,
                "latencia_seg": round(latencia, 2),
                "ram_MB_antes": round(mem_before, 1),
                "ram_MB_despues": round(mem_after, 1),
                "cpu_%_antes": cpu_before,
                "cpu_%_despues": cpu_after,
                "gpu_%_antes": gpu_before,
                "gpu_%_despues": gpu_after,
                "coherencia": coherencia,
                "longitud_respuesta": longitud_respuesta,
                "contiene_error": contiene_error
            })

            print(f"Pregunta: {pregunta}")
            print(f"Latencia: {latencia:.2f}s | RAM: {mem_after:.1f} MB | Coherencia: {coherencia} | GPU: {gpu_after}%")
            print("-" * 40)

# Guardar resultados
print("\nExportando resultados...")
df = pd.DataFrame(resultados)
df.to_csv("resultados_benchmark.csv", index=False, encoding="utf-8")
df.to_excel("resultados_benchmark.xlsx", index=False)
print("\n¡Benchmark terminado! Resultados guardados en .csv y .xlsx")

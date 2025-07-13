#!/usr/bin/env python
"""Benchmark RAG‑LLMs sobre un solo PDF y genera reportes + gráficos.

Ejemplo de uso:
    python benchmark_single_pdf.py \
        --pdf docs/mi_documento.pdf \
        --preguntas preguntas.txt

Si --pdf no se indica, toma el primer PDF encontrado en la carpeta docs/.
"""
import sys
import os
import time
import argparse

# --------- COMPROBACIÓN DE DEPENDENCIAS ANTES DE SEGUIR ---------
REQUIRED_PACKAGES = ["pandas", "psutil", "matplotlib", "openpyxl"]
missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"\n[ERROR] Faltan los siguientes paquetes: {', '.join(missing)}")
    print(f"Instala todo con:\n    pip install {' '.join(missing)}\n")
    sys.exit(1)

import pandas as pd
import psutil
from rag_core import process_pdf, get_qa_chain

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# Puedes comentar los modelos que no quieras evaluar para ahorrar tiempo
MODELOS = {
    "llama3": "llama3",
    "mistral": "mistral",
    "zephyr": "zephyr",
    "phi3": "phi3",
    "openhermes": "openhermes",
}

def cargar_preguntas(path: str) -> list[str]:
    """Lee las preguntas desde un archivo de texto"""
    if not os.path.isfile(path):
        print(f"\n[ERROR] No se encontró el archivo de preguntas: {path}\n")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def benchmark_pdf(pdf_path: str, preguntas: list[str], modelos: dict[str, str]):
    """Ejecuta el benchmark sobre un único PDF y devuelve una lista de resultados"""
    resultados = []
    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]

    # Indexar sólo la primera vez
    if not os.path.exists(f"vectorstore/{pdf_id}"):
        print(f"Indexando {pdf_id} …")
        process_pdf(pdf_path, pdf_id)

    for nombre_modelo, modelo_alias in modelos.items():
        print(f"\nEvaluando modelo: {nombre_modelo}")
        qa_chain = get_qa_chain(pdf_id, modelo_alias)

        for pregunta in preguntas:
            proceso = psutil.Process(os.getpid())
            mem_before = proceso.memory_info().rss / 1024**2  # MB
            cpu_before = proceso.cpu_percent(interval=0.1)

            if GPU_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_before = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            else:
                gpu_before = None

            prompt = (
                "Responde en español. Primero, razona paso a paso, mostrando tu pensamiento antes de responder. "
                "Al final, da la respuesta clara comenzando con: RESPUESTA FINAL: "
                f"\n\nPregunta: {pregunta}"
            )

            try:
                start = time.perf_counter()
                respuesta = qa_chain.run(prompt)
                latencia = time.perf_counter() - start
            except Exception as e:
                print(f"[ADVERTENCIA] Fallo modelo '{nombre_modelo}' con la pregunta: {pregunta}\n  Error: {e}")
                respuesta = f"[ERROR]: {e}"
                latencia = -1

            mem_after = proceso.memory_info().rss / 1024**2
            cpu_after = proceso.cpu_percent(interval=0.1)

            if GPU_AVAILABLE:
                gpu_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            else:
                gpu_after = None

            respuesta_lower = respuesta.lower() if respuesta else ""
            coherencia = 1 if "respuesta final" in respuesta_lower else 0.5 if respuesta else 0
            longitud_respuesta = len(respuesta or "")
            contiene_error = int(any(x in respuesta_lower for x in [
                "no lo sé", "no se encuentra", "no aparece", "no puedo", "no está disponible"
            ]))

            resultados.append({
                "modelo": nombre_modelo,
                "pdf": os.path.basename(pdf_path),
                "pregunta": pregunta,
                "respuesta": respuesta,
                "latencia_seg": round(latencia, 2) if latencia >= 0 else "error",
                "ram_MB_antes": round(mem_before, 1),
                "ram_MB_despues": round(mem_after, 1),
                "cpu_%_antes": cpu_before,
                "cpu_%_despues": cpu_after,
                "gpu_%_antes": gpu_before,
                "gpu_%_despues": gpu_after,
                "coherencia": coherencia,
                "longitud_respuesta": longitud_respuesta,
                "contiene_error": contiene_error,
            })

            print(f"Pregunta: {pregunta}")
            print(f"Latencia: {latencia if latencia >= 0 else 'ERROR'}s | RAM: {mem_after:.1f} MB | Coherencia: {coherencia} | GPU: {gpu_after}%")
            print("-" * 40)

    return resultados

def generar_reportes(df: pd.DataFrame, out_dir: str = "resultados"):
    """Guarda CSV y Excel con los resultados"""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "resultados_benchmark.csv")
    xlsx_path = os.path.join(out_dir, "resultados_benchmark.xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo guardar el Excel: {e}\nInstala openpyxl si aún no lo has hecho.")

    print(f"\nResultados guardados en {csv_path} y (si no hay advertencia) {xlsx_path}")

def generar_graficos(df: pd.DataFrame, out_dir: str = "resultados/plots"):
    """Genera y guarda gráficos básicos a partir del DataFrame"""
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Latencia promedio por modelo (solo valores numéricos)
    lat_prom = df[df['latencia_seg'] != "error"].groupby("modelo")["latencia_seg"].mean()
    plt.figure()
    lat_prom.sort_values().plot(kind="barh")
    plt.xlabel("Segundos")
    plt.title("Latencia promedio por modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latencia_promedio.png"))
    plt.close()

    # Coherencia promedio por modelo
    coh_prom = df.groupby("modelo")["coherencia"].mean()
    plt.figure()
    coh_prom.sort_values().plot(kind="barh")
    plt.xlabel("Score")
    plt.title("Coherencia promedio por modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coherencia_promedio.png"))
    plt.close()

    # Errores totales por modelo
    err_tot = df.groupby("modelo")["contiene_error"].sum()
    plt.figure()
    err_tot.sort_values().plot(kind="barh")
    plt.xlabel("Total de errores")
    plt.title("Errores por modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "errores_por_modelo.png"))
    plt.close()

    print(f"Gráficos guardados en {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG‑LLMs sobre un solo PDF.")
    parser.add_argument(
        "--pdf",
        help="Ruta del PDF a evaluar. Si no se indica, toma el primer PDF encontrado en docs/",
    )
    parser.add_argument(
        "--preguntas",
        default="preguntas.txt",
        help="Archivo con preguntas (uno por línea)",
    )
    args = parser.parse_args()

    # Comprobar existencia de docs/
    if not os.path.isdir("docs"):
        print("\n[ERROR] No existe la carpeta 'docs/'. Crea la carpeta y coloca tu(s) PDF(s) allí.\n")
        sys.exit(1)

    # Determinar el PDF a usar
    if args.pdf:
        if not os.path.isfile(args.pdf):
            print(f"\n[ERROR] El archivo PDF indicado no existe: {args.pdf}\n")
            sys.exit(1)
        pdf_path = args.pdf
    else:
        pdfs = [f for f in os.listdir("docs") if f.endswith(".pdf")]
        if not pdfs:
            print("\n[ERROR] No se encontró ningún PDF en la carpeta docs/\n")
            sys.exit(1)
        pdf_path = os.path.join("docs", pdfs[0])

    preguntas = cargar_preguntas(args.preguntas)
    print(f"Preguntas cargadas: {len(preguntas)}")
    print(f"PDF a evaluar: {pdf_path}\n")

    resultados = benchmark_pdf(pdf_path, preguntas, MODELOS)
    df = pd.DataFrame(resultados)

    generar_reportes(df)
    generar_graficos(df)

    print("\n¡Benchmark completado!")

if __name__ == "__main__":
    main()

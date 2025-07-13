#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark académico de modelos RAG-LLM sobre PDF: análisis de eficiencia y calidad.

Uso:
    python benchmark_single_pdf.py --pdf docs/mi_doc.pdf --preguntas preguntas.txt
"""

import sys
import os
import time
import argparse
import uuid
from datetime import datetime

# -- Revisión de dependencias --
REQUIRED_PACKAGES = ["pandas", "psutil", "matplotlib", "openpyxl", "tabulate", "seaborn"]
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
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from rag_core import process_pdf, get_qa_chain

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

MODELOS = {
    "Llama-3": "llama3",
    "Mistral": "mistral",
    "Zephyr": "zephyr",
    "Phi-3": "phi3",
    "OpenHermes": "openhermes",
}

def cargar_preguntas(path: str) -> list:
    if not os.path.isfile(path):
        print(f"\n[ERROR] No se encontró el archivo de preguntas: {path}\n")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def benchmark_pdf(pdf_path: str, preguntas: list, modelos: dict, log_file):
    resultados = []
    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Indexar solo si no existe
    if not os.path.exists(f"vectorstore/{pdf_id}"):
        print(f"Indexando {pdf_id} …")
        process_pdf(pdf_path, pdf_id)
    for nombre_modelo, modelo_alias in modelos.items():
        print(f"\n--- Evaluando modelo: {nombre_modelo} ---")
        qa_chain = get_qa_chain(pdf_id, modelo_alias)
        for i, pregunta in enumerate(preguntas, 1):
            proceso = psutil.Process(os.getpid())
            mem_before = proceso.memory_info().rss / 1024**2
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
                observacion = ""
            except Exception as e:
                print(f"[!] Fallo modelo '{nombre_modelo}' con la pregunta #{i}: {e}")
                log_file.write(f"[{nombre_modelo}] Pregunta {i} error: {e}\n")
                respuesta = f"[ERROR]: {e}"
                latencia = -1
                observacion = "Error de ejecución"
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
            if observacion == "" and latencia > 60:
                observacion = "Respuesta muy lenta (>60s)"
            resultados.append({
                "ID Corrida": run_id,
                "Fecha/Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Modelo": nombre_modelo,
                "Nombre PDF": os.path.basename(pdf_path),
                "Pregunta": pregunta,
                "Respuesta": respuesta,
                "Latencia (seg)": round(latencia, 2) if latencia >= 0 else "ERROR",
                "RAM antes (MB)": round(mem_before, 1),
                "RAM después (MB)": round(mem_after, 1),
                "CPU antes (%)": cpu_before,
                "CPU después (%)": cpu_after,
                "GPU antes (%)": gpu_before,
                "GPU después (%)": gpu_after,
                "Coherencia": coherencia,
                "Longitud respuesta": longitud_respuesta,
                "Contiene Error": contiene_error,
                "Observación": observacion
            })
            # Mostrar resumen tabla
            print(tabulate([[i, nombre_modelo, round(latencia,2) if latencia>=0 else "ERR", coherencia, contiene_error, observacion]],
                headers=["#", "Modelo", "Latencia(s)", "Coherencia", "Error", "Obs"], tablefmt="fancy_grid"))
    return resultados

def generar_reportes(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "resultados_benchmark.csv")
    xlsx_path = os.path.join(out_dir, "resultados_benchmark.xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo guardar el Excel: {e}\nInstala openpyxl si aún no lo has hecho.")
    print(f"\nResultados guardados en {csv_path} y (si no hay advertencia) {xlsx_path}")

def graficos_academicos(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    font = {'family': 'serif', 'size': 13}
    plt.rc('font', **font)
    # Boxplot de latencia
    df_lat = df[df["Latencia (seg)"] != "ERROR"].copy()
    df_lat["Latencia (seg)"] = pd.to_numeric(df_lat["Latencia (seg)"], errors="coerce")
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df_lat, x="Modelo", y="Latencia (seg)", showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black"})
    plt.title("Distribución de Latencia por Modelo")
    plt.ylabel("Latencia (segundos)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplot_latencia.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "boxplot_latencia.pdf"))
    plt.close()
    # Boxplot de coherencia
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Modelo", y="Coherencia", showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black"})
    plt.title("Distribución de Coherencia por Modelo")
    plt.ylabel("Coherencia")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplot_coherencia.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "boxplot_coherencia.pdf"))
    plt.close()
    # Heatmap de errores por modelo y pregunta
    error_matrix = df.pivot_table(index="Pregunta", columns="Modelo", values="Contiene Error", fill_value=0)
    plt.figure(figsize=(10, max(4, 0.25*len(error_matrix))))
    sns.heatmap(error_matrix, cmap="Reds", annot=True, cbar=True, linewidths=0.5, fmt='d')
    plt.title("Mapa de calor de errores por pregunta y modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_errores.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "heatmap_errores.pdf"))
    plt.close()
    print(f"Gráficos (PNG y PDF) guardados en {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark académico de modelos RAG‑LLMs sobre PDF.")
    parser.add_argument("--pdf", help="Ruta del PDF a evaluar. Si no se indica, toma el primer PDF encontrado en docs/")
    parser.add_argument("--preguntas", default="preguntas.txt", help="Archivo con preguntas (uno por línea)")
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
    print(f"\n>>> Preguntas cargadas: {len(preguntas)}")
    print(f">>> PDF a evaluar: {pdf_path}\n")
    # Carpeta única de resultados por corrida
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"resultados/benchmark_{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark.log"), "w", encoding="utf-8") as log_file:
        resultados = benchmark_pdf(pdf_path, preguntas, MODELOS, log_file)
    df = pd.DataFrame(resultados)
    generar_reportes(df, out_dir)
    graficos_academicos(df, out_dir)
    print(f"\nTodos los archivos de resultados y gráficos están en: {out_dir}\n")
    print(tabulate(df[["Modelo","Pregunta","Latencia (seg)","Coherencia","Contiene Error","Observación"]],
                   headers="keys", showindex=False, tablefmt="psql", numalign="right"))

if __name__ == "__main__":
    main()

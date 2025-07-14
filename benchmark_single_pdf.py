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
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    font = {'family': 'serif', 'size': 15}
    plt.rc('font', **font)

    # ============ VIOLINPLOTS: Distribución Latencia y Coherencia ============
    df_lat = df[df["Latencia (seg)"] != "ERROR"].copy()
    df_lat["Latencia (seg)"] = pd.to_numeric(df_lat["Latencia (seg)"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df_lat, x="Modelo", y="Latencia (seg)", inner="quartile", ax=ax)
    sns.swarmplot(data=df_lat, x="Modelo", y="Latencia (seg)", color=".3", size=4, ax=ax)
    plt.title("Distribución de latencias por modelo")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "violin_latencia.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "violin_latencia.pdf"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Modelo", y="Coherencia", inner="quartile", ax=ax)
    sns.swarmplot(data=df, x="Modelo", y="Coherencia", color=".3", size=4, ax=ax)
    plt.title("Distribución de coherencia por modelo")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "violin_coherencia.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "violin_coherencia.pdf"))
    plt.close()

    # ============ SCATTERPLOT: Latencia vs Longitud respuesta =============
    df_scatter = df_lat[df_lat["Longitud respuesta"]>0]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df_scatter, x="Longitud respuesta", y="Latencia (seg)", hue="Modelo", ax=ax)
    sns.regplot(data=df_scatter, x="Longitud respuesta", y="Latencia (seg)", scatter=False, ax=ax, color='black')
    plt.title("Relación entre longitud de respuesta y latencia")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_longitud_latencia.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "scatter_longitud_latencia.pdf"))
    plt.close()

    # ============ HEATMAP: Errores por pregunta/modelo =============
    error_matrix = df.pivot_table(index="Pregunta", columns="Modelo", values="Contiene Error", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35*len(error_matrix))))
    sns.heatmap(error_matrix.astype(int), cmap="Reds", annot=True, cbar=True, linewidths=0.5, fmt='d')
    plt.title("Mapa de calor de errores por pregunta y modelo")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_errores.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "heatmap_errores.pdf"))
    plt.close()

    # ============ RADAR CHART: Resumen multicriterio por modelo ============
    from math import pi
    summary = df.groupby("Modelo").agg({
        "Latencia (seg)": "mean",
        "Coherencia": "mean",
        "Longitud respuesta": "mean",
        "Contiene Error": "mean"
    })
    categories = summary.columns.tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    for i, row in summary.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=i)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title("Comparación multicriterio por modelo")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    fig.savefig(os.path.join(out_dir, "radar_modelos.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "radar_modelos.pdf"))
    plt.close()

    # ============ BARRAS APILADAS: Composición de errores ============
    error_comp = df.groupby(["Modelo", "Contiene Error"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9,5))
    error_comp.plot(kind="bar", stacked=True, color=["#4daf4a", "#e41a1c"], ax=ax)
    plt.title("Composición de respuestas correctas/incorrectas por modelo")
    plt.ylabel("Cantidad de respuestas")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "barras_apiladas_errores.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "barras_apiladas_errores.pdf"))
    plt.close()

    print(f"Gráficos científicos avanzados (violinplot, scatterplot, radar, heatmap, barras apiladas) guardados en {out_dir}")

def resumen_estadistico(df: pd.DataFrame, out_dir: str):
    import numpy as np

    path_txt = os.path.join(out_dir, "resumen_estadistico.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("===== RESUMEN ESTADÍSTICO POR MODELO =====\n\n")
        # Para cada modelo, mostrar resumen estadístico de cada métrica clave
        metricas = ["Latencia (seg)", "Coherencia", "Longitud respuesta"]
        for modelo, dfm in df.groupby("Modelo"):
            f.write(f"--- Modelo: {modelo} ---\n")
            for metrica in metricas:
                datos = pd.to_numeric(dfm[metrica], errors="coerce").dropna()
                if len(datos)==0: continue
                f.write(f"  {metrica}:\n")
                f.write(f"    Media: {np.mean(datos):.3f}\n")
                f.write(f"    Mediana: {np.median(datos):.3f}\n")
                f.write(f"    Desv. estándar: {np.std(datos):.3f}\n")
                f.write(f"    Mínimo: {np.min(datos):.3f}\n")
                f.write(f"    Máximo: {np.max(datos):.3f}\n")
                f.write(f"    Q1: {np.percentile(datos,25):.3f} | Q3: {np.percentile(datos,75):.3f}\n")
            # Porcentaje de errores y respuestas perfectamente coherentes
            total = len(dfm)
            errores = int(dfm["Contiene Error"].sum())
            perfectas = int((dfm["Coherencia"]==1).sum())
            f.write(f"  Respuestas con error: {errores}/{total} ({100*errores/total:.1f}%)\n")
            f.write(f"  Respuestas perfectamente coherentes: {perfectas}/{total} ({100*perfectas/total:.1f}%)\n\n")
        
        f.write("===== RANKING DE MODELOS POR MÉTRICA =====\n\n")
        # Ranking de modelos para cada métrica
        for metrica in metricas:
            resumen = df.groupby("Modelo")[metrica].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
            if metrica == "Latencia (seg)":  # menor es mejor
                resumen = resumen.sort_values()
            else:
                resumen = resumen.sort_values(ascending=False)
            f.write(f"Ranking por {metrica} (mejor primero):\n")
            for i, (modelo, valor) in enumerate(resumen.items(),1):
                f.write(f"  {i}. {modelo}: {valor:.3f}\n")
            f.write("\n")
        
        # Correlación longitud-latencia
        f.write("===== CORRELACIÓN LONGITUD-LATENCIA =====\n")
        try:
            x = pd.to_numeric(df["Longitud respuesta"], errors="coerce")
            y = pd.to_numeric(df["Latencia (seg)"], errors="coerce")
            corr = np.corrcoef(x.dropna(), y.dropna())[0,1]
            f.write(f"Correlación Pearson longitud-latencia: {corr:.3f}\n")
        except Exception as e:
            f.write(f"No se pudo calcular la correlación: {e}\n")

    print(f"Resumen estadístico exportado en {path_txt}")


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
    resumen_estadistico(df, out_dir)

    print(f"\nTodos los archivos de resultados y gráficos están en: {out_dir}\n")
    print(tabulate(df[["Modelo","Pregunta","Latencia (seg)","Coherencia","Contiene Error","Observación"]],
                   headers="keys", showindex=False, tablefmt="psql", numalign="right"))

if __name__ == "__main__":
    main()

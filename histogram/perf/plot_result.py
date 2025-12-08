import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_FILENAME = "./perf/benchmark_results.csv"
OUTPUT_GRAPH_FILENAME = "./perf/histogram_performance.png"

def create_performance_graph(input_file, output_file):

    if not os.path.exists(input_file):
        print(f"ERREUR : Le fichier d'entrée '{input_file}' est introuvable.")
        return

    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        print("ERREUR : Le fichier CSV est vide.")
        return

    for col in df.columns:
        if col != 'array_len':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    plt.figure(figsize=(12, 7))

    modes = {
        'CPU_Seq': {'label': 'CPU Séquentiel', 'color': 'blue', 'marker': 'o'},
        'CPU_OMP': {'label': 'OpenMP (Multicoeur)', 'color': 'green', 'marker': 's'},
        'GPU_CUDA': {'label': 'CUDA (GPU - GTX 1650)', 'color': 'red', 'marker': 'D'}
    }

    for mode, style in modes.items():
        if mode in df.columns:
            valid_data = df.dropna(subset=[mode])
            
            plt.plot(
                valid_data['array_len'], 
                valid_data[mode], 
                label=style['label'],
                color=style['color'], 
                marker=style['marker'],
                linestyle='-',
                linewidth=2
            )

    plt.xscale('log')
    plt.yscale('log')

    plt.title(f"Performance de l'Histogramme", fontsize=16)
    plt.xlabel("Taille du Tableau (Nombre d'éléments - Échelle Log)", fontsize=14)
    plt.ylabel("Temps d'exécution moyen (secondes - Échelle Log)", fontsize=14)
    
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(title="Mode d'exécution", loc='upper left')

    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nGraphique sauvegardé sous : {output_file}")
    print("Le graphique montre clairement l'accélération (Speedup) sur les grands ensembles de données.")
    
    # plt.show()

if __name__ == "__main__":
    create_performance_graph(INPUT_FILENAME, OUTPUT_GRAPH_FILENAME)
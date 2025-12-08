import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

INPUT_FILENAME = "./perf/block_size_optimization_results.csv"
OUTPUT_GRAPH_FILENAME = "./perf/cuda_performance_by_block.png"

def create_inverted_block_size_graph(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"ERREUR : Le fichier d'entrée '{input_file}' est introuvable.")
        print("Veuillez d'abord exécuter 'block_benchmark.py' pour générer les données.")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"ERREUR lors de la lecture du CSV: {e}")
        return

    time_cols = [col for col in df.columns if col.startswith('BS_')]
    if not time_cols:
        print("ERREUR : Aucune colonne de taille de bloc (BS_XXX) trouvée dans le CSV.")
        return
        
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    block_sizes = [int(re.search(r'BS_(\d+)', col).group(1)) for col in time_cols]
    
    plt.figure(figsize=(12, 7))

    for i, col_name in enumerate(time_cols):
        bs_value = block_sizes[i]
        
        valid_data = df[['array_len', col_name]].dropna()
        
        plt.plot(
            valid_data['array_len'],
            valid_data[col_name],
            label=f"BS: {bs_value}",
            marker='o',
            linestyle='-',
            linewidth=2
        )
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xticks(df['array_len'], [f"{a:,}" for a in df['array_len']], rotation=45, ha='right')

    plt.title("Performance CUDA vs. Taille du Tableau (Courbe par Taille de Bloc)", fontsize=16)
    plt.xlabel("Taille du Tableau (Nombre d'éléments - Échelle Log)", fontsize=14)
    plt.ylabel("Temps d'exécution moyen (secondes - Échelle Log)", fontsize=14)
    
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(title="Taille de Bloc", loc='best')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\n✅ Graphique sauvegardé sous : {output_file}")
    
if __name__ == "__main__":
    create_inverted_block_size_graph(INPUT_FILENAME, OUTPUT_GRAPH_FILENAME)
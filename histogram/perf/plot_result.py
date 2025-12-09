import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

INPUT_CSV_FILE: str = "./perf/benchmark_results.csv" 
OUTPUT_GRAPH_FILE: str = "./perf/histogram_performance.png" 

def plot_performance(csv_file: str, output_file: str):
    """
    Lit le fichier CSV de performance et génère un graphique en courbes.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[ERREUR] Le fichier CSV '{csv_file}' est introuvable. Veuillez vous assurer que le chemin est correct.")
        return

    required_cols = ['executable', 'input_size', 'average_timing']
    if not all(col in df.columns for col in required_cols):
        print(f"[ERREUR] Le fichier CSV doit contenir les colonnes : {required_cols}")
        print(f"Colonnes trouvées : {df.columns.tolist()}")
        return

    df['input_size'] = pd.to_numeric(df['input_size'])
    df['average_timing'] = pd.to_numeric(df['average_timing'])

    print(f"--- Génération du graphique de performance ---")
    print(f"Données lues depuis : {csv_file}")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    sns.lineplot(
        data=df, 
        x='input_size', 
        y='average_timing', 
        hue='executable', 
        marker='o',
        linewidth=2
    )

    plt.title('Comparaison des Performances des Implémentations d\'Histogramme', fontsize=16)
    plt.xlabel('Taille du Tableau (input_size)', fontsize=12)
    plt.ylabel('Temps d\'Exécution Moyen (secondes)', fontsize=12)
    plt.xscale('log') 
    plt.yscale('log') 

    plt.grid(True, which="both", ls="--", c='0.7')

    plt.legend(title='Exécutable', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"Le graphique a été sauvegardé sous : **{output_file}**")
    plt.show()

if __name__ == "__main__":
    plot_performance(INPUT_CSV_FILE, OUTPUT_GRAPH_FILE)
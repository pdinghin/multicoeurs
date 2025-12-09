import subprocess
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

EXECUTABLE: str = "./histogram_omp"

ARRAY_LENS: List[int] = [100000, 500000, 1000000, 5000000, 10000000]

NB_THREADS: List[int] = [1, 2, 4, 8, 12, 16, 20, 24]

NB_BINS: int = 5
NB_REPEAT: int = 10 
OUTPUT_CSV_FILE: str = "./perf/omp_scaling_data.csv"
OUTPUT_GRAPH_FILE: str = "./perf/omp_scaling_graph.png"

def run_program_and_parse(executable: str, array_len: int, nb_threads: int) -> pd.DataFrame:
    """
    Exécute le programme OpenMP avec le nombre de threads spécifié et retourne les données.
    """
    print(f"-> Exécution de {executable} (Threads: {nb_threads}, Taille: {array_len})...")
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(nb_threads)

    command: List[str] = [
        executable,
        f"--array-len", str(array_len),
        f"--nb-bins", str(NB_BINS),
        f"--nb-repeat", str(NB_REPEAT)
    ]
    
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=120,
            env=env
        )
        
        output_data: str = result.stdout.strip()
        
        if not output_data or len(output_data.split('\n')) <= 1:
            return pd.DataFrame()

        df: pd.DataFrame = pd.read_csv(io.StringIO(output_data))
        
        df['executable'] = executable
        df['nb_threads'] = nb_threads
        
        return df
        
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"   [ERREUR] Échec de l'exécution pour {nb_threads} threads: {e}")
        return pd.DataFrame()


def generate_plot(df: pd.DataFrame, output_file: str):
    """
    Génère et sauvegarde le graphique en courbes.
    """
    print(f"\n--- Génération du graphique de scaling ---")

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7)) 
    
    df['nb_threads'] = df['nb_threads'].astype('category')

    sns.lineplot(
        data=df, 
        x='input_size', 
        y='average_timing', 
        hue='nb_threads', 
        marker='o',
        linewidth=2,
    )

    plt.title(f'Scaling de {EXECUTABLE} en fonction du nombre de threads', fontsize=16)
    plt.xlabel('Taille du Tableau (input_size)', fontsize=12)
    plt.ylabel('Temps d\'Exécution Moyen (secondes)', fontsize=12)
    plt.xscale('log') 
    plt.yscale('log')

    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend(title='Nombre de Threads (OMP_NUM_THREADS)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"Le graphique a été sauvegardé sous : **{output_file}**")


def main():
    """
    Fonction principale pour orchestrer les tests et l'analyse.
    """
    all_data: List[pd.DataFrame] = []

    print("--- Début de l'analyse de scaling OpenMP ---")

    for array_len in ARRAY_LENS:
        for nb_threads in NB_THREADS:
            df_timing = run_program_and_parse(EXECUTABLE, array_len, nb_threads)
            if not df_timing.empty:
                all_data.append(df_timing)

    if not all_data:
        print("\n[FIN] Aucune donnée n'a été collectée. Vérifiez l'exécutable et les permissions.")
        return
        
    raw_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
    
    print("\n--- Analyse des données de timing (filtrage de rep=0) ---")

    filtered_df: pd.DataFrame = raw_df[raw_df['rep'] != 0].copy()
    filtered_df['timing'] = pd.to_numeric(filtered_df['timing'])
    
    average_performance: pd.DataFrame = filtered_df.groupby(['executable', 'array_len', 'nb_threads']).agg(
        average_timing=('timing', 'mean'),
        samples_for_avg=('rep', 'count') 
    ).reset_index()

    average_performance['array_len'] = pd.to_numeric(average_performance['array_len'])
    average_performance['nb_threads'] = pd.to_numeric(average_performance['nb_threads'])
    
    final_df: pd.DataFrame = average_performance.rename(columns={'array_len': 'input_size'})

    final_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n--- Succès ---")
    print(f"Les données moyennes ont été enregistrées dans **{OUTPUT_CSV_FILE}**.")
    print("Aperçu des résultats :")
    print(final_df.head().to_markdown(index=False, numalign="left"))

    generate_plot(final_df, OUTPUT_GRAPH_FILE)


if __name__ == "__main__":
    main()
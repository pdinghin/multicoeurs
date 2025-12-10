import subprocess
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# --- Configuration ---
EXECUTABLE: str = "./histogram_omp"

# Les différentes longueurs de tableau (array-len) à tester.
ARRAY_LENS: List[int] = [10**2,
    10**3,
    10**4,
    10**5,
    10**6,
    10**7  ]

# Les nombres de threads à tester (paramètre OMP_NUM_THREADS).
NB_THREADS: List[int] = [1, 2, 4, 8, 12, 16, 20, 24]

# Autres paramètres pour les exécutables
NB_BINS: int = 5
NB_REPEAT: int = 10 # Ignore rep=0, donc 9 répétitions pour la moyenne
OUTPUT_CSV_FILE: str = "omp_scaling_data.csv"
OUTPUT_GRAPH_FILE: str = "omp_scaling_graph.png"
# ---------------------

def run_program_and_parse(executable: str, array_len: int, nb_threads: int) -> pd.DataFrame:
    """
    Exécute le programme OpenMP avec le nombre de threads spécifié et retourne les données.
    """
    print(f"-> Exécution de {executable} (Threads: {nb_threads}, Taille: {array_len})...")
    
    # Définir l'environnement pour le sous-processus et définir OMP_NUM_THREADS
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
            env=env # Passage de la variable d'environnement
        )
        
        output_data: str = result.stdout.strip()
        
        if not output_data or len(output_data.split('\n')) <= 1:
            return pd.DataFrame()

        df: pd.DataFrame = pd.read_csv(io.StringIO(output_data))
        
        # Ajout des colonnes pour l'identification
        df['executable'] = executable
        df['nb_threads'] = nb_threads # Ajout du nombre de threads
        
        return df
        
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"   [ERREUR] Échec de l'exécution pour {nb_threads} threads: {e}")
        return pd.DataFrame()


def generate_plot(df: pd.DataFrame, output_file: str):
    """
    Génère et sauvegarde le graphique en courbes.
    """
    print(f"\n--- 📈 Génération du graphique de scaling ---")

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7)) 
    
    # S'assurer que 'nb_threads' est traité comme une catégorie pour la couleur
    df['nb_threads'] = df['nb_threads'].astype('category')

    # Création du graphique. Chaque valeur de 'nb_threads' aura sa propre courbe (hue).
    sns.lineplot(
        data=df, 
        x='input_size', 
        y='average_timing', 
        hue='nb_threads', 
        marker='o',
        linewidth=2,
        palette='viridis' # Choix d'une palette de couleurs
    )

    # Configuration des échelles X et Y en logarithmique
    plt.title(f'Scaling de {EXECUTABLE} en fonction du nombre de threads', fontsize=16)
    plt.xlabel('Taille du Tableau (input_size)', fontsize=12)
    plt.ylabel('Temps d\'Exécution Moyen (secondes)', fontsize=12)
    plt.xscale('log') 
    plt.yscale('log')

    import matplotlib.ticker as ticker
    ax = plt.gca() # Obtenir l'axe actuel
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=len(ARRAY_LENS))) 
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) # Pour afficher 100, 1000 etc. au lieu de 1e+02

    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend(title='Nombre de Threads (OMP_NUM_THREADS)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Sauvegarde et affichage
    plt.savefig(output_file, dpi=300)
    print(f"Le graphique a été sauvegardé sous : **{output_file}**")
    plt.show()

def main():
    """
    Fonction principale pour orchestrer les tests et l'analyse.
    """
    all_data: List[pd.DataFrame] = []

    print("--- ⏱️ Début de l'analyse de scaling OpenMP ---")

    # 1. Exécution et Collecte des Données
    for array_len in ARRAY_LENS:
        for nb_threads in NB_THREADS:
            df_timing = run_program_and_parse(EXECUTABLE, array_len, nb_threads)
            if not df_timing.empty:
                all_data.append(df_timing)

    if not all_data:
        print("\n[FIN] Aucune donnée n'a été collectée. Vérifiez l'exécutable et les permissions.")
        return
        
    raw_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
    
    # 2. Nettoyage et Analyse des Données
    print("\n--- 📊 Analyse des données de timing (filtrage de rep=0) ---")

    # Filtrage CLÉ : Ignorer la première exécution (rep=0)
    filtered_df: pd.DataFrame = raw_df[raw_df['rep'] != 0].copy()
    filtered_df['timing'] = pd.to_numeric(filtered_df['timing'])
    
    # Calculer la moyenne du timing pour chaque combinaison taille/threads
    average_performance: pd.DataFrame = filtered_df.groupby(['executable', 'input_size', 'nb_threads']).agg(
        average_timing=('timing', 'mean'),
        samples_for_avg=('rep', 'count') 
    ).reset_index()

    # Renommage et affichage
    final_df: pd.DataFrame = average_performance.rename(columns={'input_size': 'input_size'})

    # 3. Enregistrement et Génération du Graphique
    final_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n--- ✅ Succès ---")
    print(f"Les données moyennes ont été enregistrées dans **{OUTPUT_CSV_FILE}**.")
    print("🚀 Aperçu des résultats :")
    print(final_df.head().to_markdown(index=False, numalign="left"))

    generate_plot(final_df, OUTPUT_GRAPH_FILE)


if __name__ == "__main__":
    main()
import subprocess
import pandas as pd
import io
import re
from typing import List, Dict, Any

EXECUTABLES: List[str] = ["./histogram", "./histogram_omp", "./histogram_cuda"]

ARRAY_LENS: List[int] = [10**2,
    10**3,
    10**4,
    10**5,
    10**6,
    10**7]

NB_BINS: int = 5
NB_REPEAT: int = 10
OUTPUT_CSV_FILE: str = "./perf/benchmark_results.csv"

def run_program_and_parse(executable: str, array_len: int, nb_bins: int, nb_repeat: int) -> pd.DataFrame:
    """
    Exécute un programme avec les paramètres donnés et retourne ses données de timing sous forme de DataFrame.
    """
    print(f"-> Exécution de {executable} avec --array-len {array_len}...")
    
    command: List[str] = [
        executable,
        f"--array-len", str(array_len),
        f"--nb-bins", str(nb_bins),
        f"--nb-repeat", str(nb_repeat)
    ]
    
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        output_data: str = result.stdout.strip()
        
        if not output_data or len(output_data.split('\n')) <= 1:
            print(f"   [AVERTISSEMENT] Aucune donnée de timing valide n'a été trouvée pour {executable} (len={array_len}).")
            return pd.DataFrame()

        df: pd.DataFrame = pd.read_csv(io.StringIO(output_data))
        
        df['executable'] = executable
        
        return df
        
    except subprocess.CalledProcessError as e:
        print(f"   [ERREUR] Échec de l'exécution de {executable} (len={array_len}). Code d'erreur: {e.returncode}")
        print(f"   Stderr: {e.stderr.strip()}")
        return pd.DataFrame()
    except FileNotFoundError:
        print(f"   [ERREUR] Le programme {executable} est introuvable. Assurez-vous que le chemin est correct.")
        return pd.DataFrame()
    except Exception as e:
        print(f"   [ERREUR] Une erreur inattendue s'est produite lors du traitement de {executable}: {e}")
        return pd.DataFrame()

def main():
    """
    Fonction principale pour orchestrer les tests, l'analyse et l'enregistrement CSV.
    """
    all_data: List[pd.DataFrame] = []

    print("--- ⏱️ Début de la comparaison des performances ---")

    for array_len in ARRAY_LENS:
        for executable in EXECUTABLES:
            df_timing = run_program_and_parse(executable, array_len, NB_BINS, NB_REPEAT)
            if not df_timing.empty:
                all_data.append(df_timing)

    if not all_data:
        print("\n[FIN] Aucune donnée n'a été collectée. Veuillez vérifier les chemins et les permissions des exécutables.")
        return
        
    raw_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
    
    print("\n--- 📊 Analyse des données de timing ---")

    filtered_df: pd.DataFrame = raw_df[raw_df['rep'] != 0].copy()

    filtered_df['timing'] = pd.to_numeric(filtered_df['timing'])
    
    average_performance: pd.DataFrame = filtered_df.groupby(['executable', 'array_len']).agg(
        average_timing=('timing', 'mean'),
        nb_bins=('nb_bins', 'first'),
        nb_repeat=('nb_repeat', 'first')
    ).reset_index()

    final_df: pd.DataFrame = average_performance[[
        'executable', 
        'array_len', 
        'nb_bins', 
        'nb_repeat', 
        'average_timing'
    ]]
    
    final_df.rename(columns={'array_len': 'input_size'}, inplace=True)

    final_df.to_csv(OUTPUT_CSV_FILE, index=False)

    print(f"\n--- Succès ---")
    print(f"Les résultats de performance moyenne ont été enregistrés dans **{OUTPUT_CSV_FILE}**.")
    print("\nAperçu des résultats finaux (les 5 premières lignes) :")
    print(final_df.head().to_markdown(index=False, numalign="left"))


if __name__ == "__main__":
    main()
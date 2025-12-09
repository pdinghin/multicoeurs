import subprocess
import csv
import io
import time


EXECUTABLES = {
    "CPU_Seq": "./histogram",
    "CPU_OMP": "./histogram_omp",
    "GPU_CUDA": "./histogram_cuda"
}

ARRAY_LENGTHS = [
    10**2,
    10**3,
    10**4,
    10**5,
    10**6,
    10**7    
]

NB_BINS = 1024
NB_REPEAT = 6

OUTPUT_FILENAME = "./perf/benchmark_results.csv"


def run_benchmark(executable, array_len, nb_bins, nb_repeat):
    """
    Exécute un programme, retourne le temps moyen des (nb_repeat - 1) dernières itérations.
    """
    command = [
        executable,
        "--array-len", str(array_len),
        "--nb-bins", str(nb_bins),
        "--nb-repeat", str(nb_repeat)
    ]
    
    print(f"  -> Exécution: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        csv_data = result.stdout.strip()
        f = io.StringIO(csv_data)
        reader = csv.DictReader(f)
        
        timings = []
        check_failures = 0

        for row in reader:
            try:
                timing = float(row['timing'])
                timings.append(timing)
                
                if int(row['check_status']) != 0:
                    check_failures += 1
            except (ValueError, KeyError):
                continue 

        if not timings or len(timings) < nb_repeat:
            print(f"    [ATTENTION] Seulement {len(timings)} temps trouvés. Attendu: {nb_repeat}.")
            return None, 0
            
        warm_up_time = timings[0]
        measured_timings = timings[1:]
        print(measured_timings)
        
        
        if check_failures > 0:
            print(f"    [ÉCHEC] {check_failures} répétitions ont échoué à la vérification.")
            
        avg_time = sum(measured_timings) / len(measured_timings)
        
        print(f"    [Warm-up] Temps du premier run (ignoré): {warm_up_time:.6f} s")

        return avg_time, check_failures

    except subprocess.CalledProcessError as e:
        print(f"    [ERREUR] Programme '{executable}' a retourné le code {e.returncode}.")
        print(f"    Sortie d'erreur (stderr):\n{e.stderr}")
        return None, 1
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Le programme a dépassé le temps maximum alloué.")
        return None, 1
    except FileNotFoundError:
        print(f"    [ERREUR] Exécutable '{executable}' non trouvé. Assurez-vous d'avoir fait 'make'.")
        return None, 1

def main():
    print("Début de l'évaluation avec pré-chauffage (Warm-up).")
    print(f"-> Chaque test est répété {NB_REPEAT} fois. La première mesure est ignorée.")
    
    all_results = []
    fieldnames = ['array_len'] + list(EXECUTABLES.keys())

    for array_len in ARRAY_LENGTHS:
        print(f"\n--- Taille du Tableau: {array_len:,} ---")
        
        current_result = {'array_len': array_len}
        
        for name, executable in EXECUTABLES.items():
            
            print(f"  [MODE: {name}]")
            
            avg_time, failures = run_benchmark(executable, array_len, NB_BINS, NB_REPEAT)
            
            if avg_time is not None:
                current_result[name] = avg_time
                print(f"  -> Temps moyen (5 runs mesurés): {avg_time:.6f} secondes")
            else:
                current_result[name] = "N/A"
                
            time.sleep(1) 
            
        all_results.append(current_result)

    try:
        with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nTerminé. Résultats sauvegardés dans {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"\n[FATAL] Erreur lors de l'écriture du fichier CSV: {e}")

if __name__ == "__main__":
    main()
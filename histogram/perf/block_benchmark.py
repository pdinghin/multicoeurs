import subprocess
import csv
import io
import time
import os

CUDA_SOURCE_FILE = "./histogram_cuda.cu"
EXECUTABLE_PREFIX = "./histogram_cuda" 

BLOCK_SIZES = [
    64,
    128,
    256,
    512,
    1024
]

ARRAY_LENGTHS = [
    10**2,
    10**3,
    10**4,
    10**5,
    10**6,
    10**7,
    5 * 10**7
]

NB_BINS = 1024
NB_REPEAT = 6
ARCH_FLAG = "sm_86"

OUTPUT_FILENAME = "./perf/block_size_optimization_results.csv"


def compile_cuda(block_size):
    """ Compile le source CUDA en définissant BLOCK_SIZE via l'option -D. """
    exe_name = f"{EXECUTABLE_PREFIX}_{block_size}"
    
    compile_command = [
        "nvcc",
        f"-DBLOCK_SIZE={block_size}",
        f"-arch={ARCH_FLAG}",
        "-g", "-O3",
        CUDA_SOURCE_FILE,
        "-o", exe_name,
        "-lcudart", "-lm"
    ]
    
    print(f"\nCompilation pour BLOCK_SIZE={block_size}...")
    try:
        subprocess.run(compile_command, check=True, capture_output=True)
        return exe_name
    except subprocess.CalledProcessError as e:
        print(f"ERREUR DE COMPILATION pour BLOCK_SIZE={block_size}:")
        print(e.stderr.decode())
        return None
    except FileNotFoundError:
        print("ERREUR : Le compilateur 'nvcc' ou le fichier source est introuvable.")
        return None

def run_benchmark(executable, array_len, nb_bins, nb_repeat, block_size):
    """ Exécute le programme et retourne le temps moyen des 5 dernières itérations. """
    command = [
        executable,
        "--array-len", str(array_len),
        "--nb-bins", str(nb_bins),
        "--nb-repeat", str(nb_repeat)
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True, 
            timeout=120
        )
        
        f = io.StringIO(result.stdout.strip())
        reader = csv.DictReader(f)
        
        timings = []
        for row in reader:
            try:
                timings.append(float(row['timing']))
            except (ValueError, KeyError):
                continue
        
        if len(timings) < nb_repeat:
            return None
            
        measured_timings = timings[1:] 
        avg_time = sum(measured_timings) / len(measured_timings)
        
        print(f"  -> Résultat (BS={block_size}, ArrLen={array_len:,}): {avg_time:.6f} s")
        return avg_time

    except subprocess.CalledProcessError:
        return "ERROR"
    except Exception:
        return "N/A"

def cleanup(executables):
    """ Supprime tous les exécutables temporaires créés. """
    for exe in executables:
        if os.path.exists(exe):
            os.remove(exe)
    print("\n Nettoyage des exécutables temporaires terminé.")

def main():
    print(" Début de l'optimisation de la taille de bloc CUDA...")
    print(f"-> Architecture cible : {ARCH_FLAG}")
    
    executables_created = {}
    
    # 1. Compilation de toutes les versions
    for bs in BLOCK_SIZES:
        exe_path = compile_cuda(bs)
        if exe_path:
            executables_created[bs] = exe_path
        
    if not executables_created:
        print("\nAucun exécutable CUDA n'a pu être compilé. Arrêt du benchmark.")
        return

    # 2. Exécution des tests et stockage des résultats
    all_results = []
    
    # En-têtes du fichier de sortie : Taille du Tableau + une colonne par taille de bloc
    fieldnames = ['array_len'] + [f"BS_{bs}" for bs in BLOCK_SIZES]

    for array_len in ARRAY_LENGTHS:
        print(f"\n--- TEST : Taille du Tableau = {array_len:,} ---")
        current_result = {'array_len': array_len}
        
        for bs in BLOCK_SIZES:
            if bs in executables_created:
                exe_path = executables_created[bs]
                
                start_time = time.time()
                avg_time = run_benchmark(exe_path, array_len, NB_BINS, NB_REPEAT, bs)
                
                current_result[f"BS_{bs}"] = avg_time
                
                # Petite pause pour stabiliser le GPU entre les tests
                time.sleep(0.5) 
            else:
                current_result[f"BS_{bs}"] = "Compil. Error"
                
        all_results.append(current_result)

    try:
        with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nTerminé. Résultats de l'optimisation sauvegardés dans {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"\n[FATAL] Erreur lors de l'écriture du fichier CSV: {e}")
        
    cleanup(executables_created.values())

if __name__ == "__main__":
    main()
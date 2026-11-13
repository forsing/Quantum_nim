# Quantum_nim



import random
import numpy as np
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



def get_nim_features(numbers):
    nim_sum = 0
    for num in numbers:
        nim_sum ^= int(num)
    parity_vector = []
    temp = nim_sum
    for _ in range(6):
        parity_vector.append(temp & 1)
        temp >>= 1
    parity_vector.reverse()
    return parity_vector, nim_sum

def run_quantum_nim_circuit(parity_vector, nim_sum):
    qc = QuantumCircuit(6)
    for idx, bit in enumerate(parity_vector):
        if bit == 1:
            qc.x(idx)
    rotation_angle = float((nim_sum % 360) * np.pi / 180.0)
    for i in range(6):
        qc.ry(rotation_angle, i)
    for i in range(5):
        qc.cx(i, i + 1)
    state = Statevector(qc)
    amplitudes = state.data
    probabilities = np.array(np.abs(amplitudes) ** 2, dtype=np.float64)
    norm = np.sum(probabilities)
    if norm > 0:
        probabilities = probabilities / norm
    return probabilities

def predict_next_combination(filepath):
    if not filepath.exists():
        return None, 0.0, 0.0
    draws = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                try:
                    draws.append([int(x) for x in parts[:7]])
                except ValueError:
                    continue
    if len(draws) < 15:
        return None, 0.0, 0.0
    
    total_transitions = len(draws) - 1
    train_size = int(total_transitions * 0.7)
    val_size = int(total_transitions * 0.15)
    
    all_features = []
    for i in range(len(draws)):
        p_vec, n_sum = get_nim_features(draws[i])
        amps = run_quantum_nim_circuit(p_vec, n_sum)
        all_features.append(amps)
    all_features = np.array(all_features, dtype=np.float64)
    
    X = all_features[:-1]
    
    target_matrix = np.zeros((len(draws) - 1, 39), dtype=np.float64)
    for idx in range(len(draws) - 1):
        for num in draws[idx + 1]:
            if 1 <= num <= 39:
                target_matrix[idx, num - 1] = 1.0
                
    X_train = X[:train_size]
    y_train = target_matrix[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = target_matrix[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = target_matrix[train_size + val_size:]
    
    weights_train = np.linalg.pinv(X_train).dot(y_train)
    
    val_loss = 0.0
    if len(X_val) > 0:
        val_preds = X_val.dot(weights_train)
        for idx, pred in enumerate(val_preds):
            actual_indices = np.where(y_val[idx] == 1.0)[0] + 1
            actual_nums = set([int(x) for x in actual_indices])
            pred_indices = np.argsort(pred)[-7:] + 1
            pred_nums = set([int(x) for x in pred_indices])
            matches = len(actual_nums.intersection(pred_nums))
            val_loss += (7 - matches)
        val_loss /= len(X_val)
        
    test_loss = 0.0
    if len(X_test) > 0:
        test_preds = X_test.dot(weights_train)
        for idx, pred in enumerate(test_preds):
            actual_indices = np.where(y_test[idx] == 1.0)[0] + 1
            actual_nums = set([int(x) for x in actual_indices])
            pred_indices = np.argsort(pred)[-7:] + 1
            pred_nums = set([int(x) for x in pred_indices])
            matches = len(actual_nums.intersection(pred_nums))
            test_loss += (7 - matches)
        test_loss /= len(X_test)
        
    final_weights = np.linalg.pinv(X).dot(target_matrix)
    last_features = all_features[-1]
    next_distribution = last_features.dot(final_weights)
    
    next_indices = np.argsort(next_distribution)[-7:]
    next_combination = sorted([int(idx + 1) for idx in next_indices])
    
    return next_combination, float(val_loss), float(test_loss)

def main():
    print("Metoda: The Secret of Nim: Game-Theoretic State Invariants")
    print("Posebnost: Kvantno modeliranje stabilnih stanja kroz 6-bitnu Nim-sumu")
    print("----------------------------------------------------------------------")
    for path in FILES:
        name = path.stem
        next_comb, val_metric, test_metric = predict_next_combination(path)
        if next_comb is not None:
            print(f"Fajl: {name}")
            print(f"Validacioni gubici: {val_metric:.4f}")
            print(f"Test gubici: {test_metric:.4f}")
            print(f"NEXT: {next_comb}")
        else:
            print(f"Fajl: {name} nema dovoljno podataka.")

if __name__ == "__main__":
    main()



"""
Metoda: The Secret of Nim: Game-Theoretic State Invariants
Posebnost: Kvantno modeliranje stabilnih stanja kroz 6-bitnu Nim-sumu
----------------------------------------------------------------------

Fajl: loto7_4718_k90
Validacioni gubici: 5.6860
Test gubici: 5.7292
NEXT: [3, 7, 12, 20, 21, 22, 23]

Fajl: loto7_4718_k90_loto_2981
Validacioni gubici: 5.7651
Test gubici: 5.7740
NEXT: [21, 22, 25, 26, 29, 30, 31]

Fajl: loto7_4718_k90_loto_plus_1737
Validacioni gubici: 5.7308
Test gubici: 5.7969
NEXT: [2, 5, 7, 8, 10, 23, 27]
"""



"""
Nim 
A Mathematical Game of Strategy

The Secret of Nim: Game-Theoretic State Invariants. 
Metoda koristi matematičku teoriju igara 
(Nim-sumu, odnosno ekskluzivnu disjunkciju XOR nad binarnim reprezentacijama gomila) 
za otkrivanje stabilnih invarijanti unutar numeričkih nizova, 
umesto standardnih vremenskih serija.

Loto kombinaciju mogu predstaviti kao trenutno stanje gomila u igri Nim. 
Izračunavanjem Nim-sume (\(\bigoplus \)) istorijskih prelaza, algoritam uči stabilna kvantna stanja unutar prostora od 39 brojeva, identifikujući dobitne pozicije (safe/zero states) i preslikavajući ih kroz amplitude 6-kubitnog registra.

• Metoda: The Secret of Nim: Game-Theoretic State Invariants Quantum Circuit.
• Kvantno kodiranje: Istorijska kola se prevode u stanja Nim igre, gde se izračunava ukupna Nim-suma (XOR) i pojedinačni doprinosi. Prvih 6 bitova te Nim-sume mapira se u bazna kvantna stanja 6-kubitnog registra (6 qubita = 64 stanja).
"""

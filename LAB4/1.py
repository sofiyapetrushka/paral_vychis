import math
import numpy as np
import pandas as pd

def strong_time_model(T1, procs, p, comm_base, comm_per_level, jitter=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    compute = T1 * ((1 - p) + p / procs)
    comm = (comm_base + comm_per_level * math.log2(procs))
    noise = 1.0 + rng.normal(0.0, jitter)  
    t = (compute + comm) * max(0.85, noise)
    return t, compute, comm

def weak_params(base_N, base_M, procs, base_procs=2):
    scale = procs / base_procs
    N = base_N  
    M = int(base_M * scale)
    return N, M

def estimate_bytes(task, N, M, procs, scaling):
    vec_bytes = N * 8
    if task in ("lr1",):
        k = 2
    elif task in ("lr2",):
        k = 4
    elif task in ("lr3_1",):
        k = 6
    else:  
        k = 5

    bytes_sent = k * vec_bytes * (procs - 1) * (1 + 0.05 * math.log2(procs))
    bytes_sent *= (1 + min(0.3, M / 2e7 * 0.1))  
    return int(bytes_sent)

def main():
    rng = np.random.default_rng(123)

    datasets = {
        "A": {"N": 200,  "M": 20_000_000},
        "B": {"N": 500,  "M": 8_000_000},
        "C": {"N": 1000, "M": 2_000_000},
    }

    tasks = ["lr1", "lr2", "lr3_1", "lr3_2"]

    strong_procs = [1, 2, 4, 8, 16, 32, 64]   
    weak_procs = [1, 2, 4, 8, 16]            
    rows = []

    base_T1 = {
        "A": {"lr1": 18.0, "lr2": 15.0, "lr3_1": 12.0, "lr3_2": 11.0},
        "B": {"lr1": 20.0, "lr2": 16.5, "lr3_1": 13.5, "lr3_2": 12.5},
        "C": {"lr1": 22.0, "lr2": 18.0, "lr3_1": 15.0, "lr3_2": 13.8},
    }

    p_map = {"lr1": 0.90, "lr2": 0.93, "lr3_1": 0.95, "lr3_2": 0.965}

    comm_params = {
        "lr1":   (0.10, 0.06),  
        "lr2":   (0.12, 0.07),
        "lr3_1": (0.18, 0.11),  
        "lr3_2": (0.14, 0.08), 
    }

    for ds, prm in datasets.items():
        N, M = prm["N"], prm["M"]
        for task in tasks:
            T1 = base_T1[ds][task]
            p = p_map[task]
            comm_base, comm_per_level = comm_params[task]

            for procs in strong_procs:
                t, comp, comm = strong_time_model(
                    T1=T1,
                    procs=procs,
                    p=p,
                    comm_base=comm_base,
                    comm_per_level=comm_per_level,
                    jitter=0.02,
                    rng=rng
                )

                rows.append({
                    "scaling": "strong",
                    "dataset": ds,
                    "task": task,
                    "N": N,
                    "M": M,
                    "procs": procs,
                    "time_sec": round(float(t), 6),
                    "comp_sec": round(float(comp), 6),
                    "comm_sec": round(float(comm), 6),
                    "bytes_sent": estimate_bytes(task, N, M, procs, "strong")
                })

    base_N, base_M = datasets["A"]["N"], datasets["A"]["M"]

    for task in tasks:
        p = p_map[task] - 0.02
        comm_base, comm_per_level = comm_params[task]
        comm_base *= 1.2
        comm_per_level *= 1.25

        base_T1 = 18.0 if task == "lr1" else 15.0 if task == "lr2" else 12.0 if task == "lr3_1" else 11.0

        for procs in weak_procs:
            N, M = weak_params(base_N, base_M, procs, base_procs=2)

            T1_scaled = base_T1 * (M / base_M)

            t, comp, comm = strong_time_model(
                T1=T1_scaled,
                procs=procs,
                p=p,
                comm_base=comm_base,
                comm_per_level=comm_per_level,
                jitter=0.02,
                rng=rng
            )

            rows.append({
                "scaling": "weak",
                "dataset": "W",     
                "task": task,
                "N": N,
                "M": M,
                "procs": procs,
                "time_sec": round(float(t), 6),
                "comp_sec": round(float(comp), 6),
                "comm_sec": round(float(comm), 6),
                "bytes_sent": estimate_bytes(task, N, M, procs, "weak")
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["scaling", "dataset", "task", "procs"]).reset_index(drop=True)
    df.to_csv("lab4_results.csv", index=False)
    print("OK: wrote lab4_results.csv")
    print(df.head(10))

if __name__ == "__main__":
    main()
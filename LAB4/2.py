import matplotlib
matplotlib.use("Agg")  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("lab4_results.csv")

    for (dataset, task), g in df.groupby(["dataset", "task"]):
        g = g.sort_values("procs")
        procs = g["procs"].to_numpy()
        t = g["time_sec"].to_numpy()

        if (procs == 1).any():
            t1 = float(g.loc[g["procs"] == 1, "time_sec"].iloc[0])
        else:
            t1 = float(t[0])

        speedup = t1 / t
        eff = speedup / procs

        plt.figure()
        plt.plot(procs, t, marker="o")
        plt.xlabel("Processes")
        plt.ylabel("Time (s)")
        plt.title(f"Time: {task}, dataset {dataset}")
        plt.grid(True)
        plt.savefig(f"time_{task}_{dataset}.png", dpi=150)
        plt.close()  

        plt.figure()
        plt.plot(procs, speedup, marker="o")
        plt.xlabel("Processes")
        plt.ylabel("Speedup")
        plt.title(f"Speedup: {task}, dataset {dataset}")
        plt.grid(True)
        plt.savefig(f"speedup_{task}_{dataset}.png", dpi=150)

        plt.figure()
        plt.plot(procs, eff, marker="o")
        plt.xlabel("Processes")
        plt.ylabel("Efficiency")
        plt.title(f"Efficiency: {task}, dataset {dataset}")
        plt.grid(True)
        plt.savefig(f"eff_{task}_{dataset}.png", dpi=150)
        plt.close()  

if __name__ == "__main__":
    main()
from mpi4py import MPI
import numpy as np
import time
import os

def cg_hybrid(N=5000, maxit=300):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))

    nloc = N // size
    x = np.zeros(nloc)
    b = np.ones(nloc)

    r = b.copy()
    p = r.copy()

    rr = comm.allreduce(np.dot(r, r), op=MPI.SUM)

    t0 = time.time()

    for _ in range(maxit):
        # BLAS -> OpenMP
        Ap = np.dot(np.eye(nloc), p)
        pAp = comm.allreduce(np.dot(p, Ap), op=MPI.SUM)
        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap
        rr_new = comm.allreduce(np.dot(r, r), op=MPI.SUM)
        if rr_new < 1e-8:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    t1 = time.time()
    if rank == 0:
        print(f"[CG HYBRID] time={t1 - t0:.4f} s")

if __name__ == "__main__":
    cg_hybrid()
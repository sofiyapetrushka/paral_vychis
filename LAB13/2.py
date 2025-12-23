from mpi4py import MPI
import numpy as np

def cg_async(N=5000, maxit=300):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nloc = N // size
    x = np.zeros(nloc)
    b = np.ones(nloc)

    r = b.copy()
    p = r.copy()

    rr = comm.allreduce(np.dot(r, r), op=MPI.SUM)

    for _ in range(maxit):
        Ap = p.copy()

        # асинхронная редукция
        req = comm.Iallreduce(np.dot(p, Ap), op=MPI.SUM)
        pAp = req.wait()

        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap

        rr_new = comm.allreduce(np.dot(r, r), op=MPI.SUM)
        if rr_new < 1e-8:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    if rank == 0:
        print("[CG ASYNC] done")

if __name__ == "__main__":
    cg_async()
import time
import numpy as np
from utils import make_grid, u_init, u_left, u_right, step_explicit

def solve_seq(eps=10**(-1.5), N=800, M=300_000, save_every=50_000):
    a, b = 0.0, 1.0
    t0, T = 0.0, 6.0

    x, t, h, tau = make_grid(a, b, t0, T, N, M)

    u_prev = np.array([u_init(xi) for xi in x], dtype=float)
    # выставим границы для t0
    u_prev[0] = u_left(t0)
    u_prev[-1] = u_right(t0)

    snaps = []  # для верификации (не все слои!)
    snaps_t = []

    start = time.perf_counter()
    for m in range(M):
        u_next = step_explicit(u_prev, eps, tau, h)
        tm1 = t[m+1]
        u_next[0] = u_left(tm1)
        u_next[-1] = u_right(tm1)

        if save_every and (m % save_every == 0 or m == M-1):
            snaps.append(u_next.copy())
            snaps_t.append(tm1)

        u_prev = u_next
    elapsed = time.perf_counter() - start

    return x, np.array(snaps), np.array(snaps_t), elapsed

if __name__ == "__main__":
    x, snaps, snaps_t, elapsed = solve_seq()
    print(f"[SEQ] done. saved_snaps={len(snaps)} time={elapsed:.6f} s")
    print("u(t_end)[0], u(t_end)[mid], u(t_end)[-1] =",
          snaps[-1][0], snaps[-1][len(x)//2], snaps[-1][-1])
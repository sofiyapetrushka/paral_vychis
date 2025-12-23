from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
import os

N, M = 800, 300000
a, b = 0, 1
t0, T = 0, 6
eps = 10**(-1.5)

def u_init(x):
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t):
    return -1.0

def u_right(t):
    return 1.0

def solve_sequential():
    #Последовательная версия
    h = (b - a) / N
    tau = (T - t0) / M
    x = np.linspace(a, b, N + 1)
    u = u_init(x)
    u_next = np.zeros_like(u)
    
    start = time.time()
    
    for m in range(M):
        t = t0 + (m + 1) * tau
        u_next[0] = u_left(t)
        u_next[-1] = u_right(t)
        
        for i in range(1, N):
            d2 = (u[i+1] - 2*u[i] + u[i-1]) / h**2
            d1 = (u[i+1] - u[i-1]) / (2*h)
            u_next[i] = u[i] + eps*tau*d2 + tau*u[i]*d1 + tau*u[i]**3
        
        u[:] = u_next[:]
    
    return time.time() - start, x, u

class ParallelSolver:
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        self.h = (b - a) / N
        self.tau = (T - t0) / M
        
        # Распределение сетки
        local_N = (N - 1) // self.size
        self.local_n = local_N + 2
        
        start_idx = self.rank * local_N
        self.local_x = np.linspace(a + start_idx*self.h, 
                                  a + (start_idx + local_N + 1)*self.h, 
                                  self.local_n)
        
        self.u = np.zeros(self.local_n)
        self.u_next = np.zeros(self.local_n)
        
        # Буферы обмена
        self.send_buf = np.zeros(2)
        self.recv_buf = np.zeros(2)
    
    def initialize(self):
        self.u[:] = u_init(self.local_x)
        self.exchange_ghost()
        
        if self.rank == 0:
            self.u[1] = u_left(t0)
        if self.rank == self.size - 1:
            self.u[-2] = u_right(t0)
    
    def exchange_ghost(self):
        left = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        right = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL
        
        self.send_buf[0] = self.u[1]      # левая граница
        self.send_buf[1] = self.u[-2]     # правая граница
        
        # Обмен с соседями
        self.comm.Sendrecv([self.send_buf[0:1], MPI.DOUBLE], left, 0,
                          [self.recv_buf[1:2], MPI.DOUBLE], right, 0)
        self.comm.Sendrecv([self.send_buf[1:2], MPI.DOUBLE], right, 1,
                          [self.recv_buf[0:1], MPI.DOUBLE], left, 1)
        
        # Обновление ghost cells
        if left != MPI.PROC_NULL:
            self.u[0] = self.recv_buf[0]
        if right != MPI.PROC_NULL:
            self.u[-1] = self.recv_buf[1]
    
    def time_step(self, t):
        #Шаг по времени
        if self.rank == 0:
            self.u[1] = u_left(t)
        if self.rank == self.size - 1:
            self.u[-2] = u_right(t)
        
        for i in range(1, self.local_n - 1):
            d2 = (self.u[i+1] - 2*self.u[i] + self.u[i-1]) / self.h**2
            d1 = (self.u[i+1] - self.u[i-1]) / (2*self.h)
            self.u_next[i] = self.u[i] + eps*self.tau*d2 + self.tau*self.u[i]*d1 + self.tau*self.u[i]**3
        
        self.u[:] = self.u_next[:]
        self.exchange_ghost()
    
    def solve(self):
        self.initialize()
        
        start = MPI.Wtime()
        for m in range(M):
            t = t0 + (m + 1) * self.tau
            self.time_step(t)
        
        self.comm.barrier()
        return MPI.Wtime() - start
    
    def gather_solution(self):
        #Сбор решения на процессе 0
        local_data = self.u[1:-1]
        local_size = len(local_data)
        
        sizes = np.zeros(self.size, dtype=int)
        self.comm.Allgather([np.array([local_size]), MPI.INT], 
                           [sizes, MPI.INT])
        
        displs = np.cumsum(np.append(0, sizes[:-1]))
        
        if self.rank == 0:
            global_u = np.zeros(np.sum(sizes))
        else:
            global_u = None
        
        self.comm.Gatherv([local_data, MPI.DOUBLE],
                         [global_u, sizes, displs, MPI.DOUBLE] if self.rank == 0 else None,
                         root=0)
        
        return global_u

def run_benchmark():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Параллельное решение уравнения теплопроводности")
        print(f"Параметры: N={N}, M={M}")
        print("-" * 50)
        
        # Последовательная версия
        seq_time, seq_x, seq_u = solve_sequential()
        print(f"Последовательная: {seq_time:.2f} сек")
        
        # Сохранение результатов
        os.makedirs('results', exist_ok=True)
        np.savez('results/seq_solution.npz', x=seq_x, u=seq_u, time=seq_time)
    
    comm.barrier()
    
    # Параллельная версия
    solver = ParallelSolver(comm)
    par_time = solver.solve()
    
    # Сбор результатов
    if rank == 0:
        process_counts = comm.Get_size()
        
        # Сбор времени от всех процессов
        times = comm.gather(par_time, root=0)
        max_time = max(times) if times else par_time
        
        print(f"Параллельная ({process_counts} процессов): {max_time:.2f} сек")
        print(f"Ускорение: {seq_time/max_time:.2f}x")
        print(f"Эффективность: {seq_time/(max_time*process_counts):.1%}")
    
    # Сбор решения для проверки
    if rank == 0:
        global_u = solver.gather_solution()
        
        # Загрузка последовательного решения
        seq_data = np.load('results/seq_solution.npz')
        seq_u_inner = seq_data['u'][1:-1]  # Внутренние точки
        
        # Сравнение
        error = np.max(np.abs(global_u - seq_u_inner))
        print(f"Ошибка: {error:.2e}")
        
        # Сохранение параллельного решения
        x_full = np.linspace(a, b, N + 1)
        u_full = np.zeros(N + 1)
        u_full[0] = u_left(T)
        u_full[-1] = u_right(T)
        u_full[1:-1] = global_u
        
        np.savez(f'results/par_solution_p{comm.Get_size()}.npz', 
                x=x_full, u=u_full, time=par_time)
        
        return seq_time, max_time, error
    
    return None, None, None

def plot_results():
    # Сбор данных для разного числа процессов
    seq_data = np.load('results/seq_solution.npz')
    seq_time = seq_data['time']
    
    processes = [1, 2, 4, 8, 16]
    times = [seq_time]
    speedups = [1.0]
    errors = [0.0]
    
    for p in processes[1:]:
        try:
            par_data = np.load(f'results/par_solution_p{p}.npz')
            times.append(par_data['time'])
            speedups.append(seq_time / par_data['time'])
            
            # Вычисление ошибки
            seq_u = seq_data['u'][1:-1]
            par_u = par_data['u'][1:-1]
            errors.append(np.max(np.abs(seq_u - par_u)))
        except:
            times.append(np.nan)
            speedups.append(np.nan)
            errors.append(np.nan)
    
    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График времени
    ax1.plot(processes, times, 'bo-')
    ax1.set_xlabel('Процессы')
    ax1.set_ylabel('Время (с)')
    ax1.set_title('Время выполнения')
    ax1.grid(True)
    
    # График ускорения
    ax2.plot(processes, speedups, 'ro-', label='Реальное')
    ax2.plot(processes, processes, 'k--', label='Идеальное')
    ax2.set_xlabel('Процессы')
    ax2.set_ylabel('Ускорение')
    ax2.set_title('Ускорение')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/performance.png', dpi=150)
    plt.show()
    
    print("\nРезультаты:")
    print(f"{'Процессы':<10} {'Время(с)':<12} {'Ускорение':<12} {'Ошибка':<12}")
    print("-" * 50)
    for p, t, s, e in zip(processes, times, speedups, errors):
        print(f"{p:<10} {t:<12.2f} {s:<12.2f} {e:<12.2e}")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        run_benchmark()
        plot_results()
    else:
        run_benchmark()
    
    MPI.Finalize()
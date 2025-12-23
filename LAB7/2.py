from mpi4py import MPI
import numpy as np

def thomas_solve(a, b, c, d):
    """Последовательный метод прогонки для трёхдиагональной системы.
    a — поддиагональ (длина n-1)
    b — диагональ (длина n)
    c — наддиагональ (длина n-1)
    d — правая часть (длина n)
    """
    n = len(b)
    a = np.concatenate(([0.0], a))  
    c = np.concatenate((c, [0.0]))

    # Прямой ход
    for i in range(1, n):
        w = a[i] / b[i-1]
        b[i] = b[i] - w * c[i-1]
        d[i] = d[i] - w * d[i-1]

    # Обратный ход
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Параметры глобальной системы 
    N = 100  # Общий размер системы 
    if rank == 0:
        # Пример: система с диагональным преобладанием
        a_global = np.full(N-1, -1.0, dtype=np.float64)  # поддиагональ
        b_global = np.full(N, 2.0, dtype=np.float64)     # диагональ
        c_global = np.full(N-1, -1.0, dtype=np.float64)  # наддиагональ
        d_global = np.random.rand(N).astype(np.float64)

        # Известное решение для проверки (опционально)
        # x_true = np.sin(np.pi * np.arange(1, N+1) / (N+1))
        # d_global = (2 * x_true) - np.concatenate(([0], x_true[:-1])) - np.concatenate((x_true[1:], [0]))
    else:
        a_global = b_global = c_global = d_global = None

    # Разделение на блоки: каждый процесс получает непрерывный блок строк
    # Вычисляем размеры блоков
    base_size = N // size
    remainder = N % size
    local_n = base_size + (1 if rank < remainder else 0)

    # Смещения и размеры
    counts = []
    displs = []
    s = 0
    for r in range(size):
        cnt = base_size + (1 if r < remainder else 0)
        counts.append(cnt)
        displs.append(s)
        s += cnt

    local_start = displs[rank]
    local_end = local_start + local_n

    # Рассылаем локальные части
    b = np.empty(local_n, dtype=np.float64)  # диагональ
    d = np.empty(local_n, dtype=np.float64)  # правая часть

    comm.Scatterv([b_global, counts, displs, MPI.DOUBLE], b, root=0)
    comm.Scatterv([d_global, counts, displs, MPI.DOUBLE], d, root=0)

    # Поддиагональ a: длина N-1 → принадлежит строке i (i от 1 до N-1)
    # Наддиагональ c: длина N-1 → принадлежит строке i (i от 0 до N-2)
    # Локальные поддиагонали и наддиагонали:
    a_local = np.empty(local_n, dtype=np.float64)  # a_local[0] не используется (кроме связи!)
    c_local = np.empty(local_n, dtype=np.float64)  # c_local[-1] не используется

    # Заполняем a_local и c_local
    # Для внутренних узлов — просто -1
    a_local[:] = -1.0
    c_local[:] = -1.0

    # Корректируем граничные случаи
    if local_start == 0:
        a_local[0] = 0.0  # первая строка не имеет поддиагонали
    if local_end == N:
        c_local[-1] = 0.0  # последняя строка не имеет наддиагонали

    # Этап 2.2: Локальный прямой ход
    # Выполняем локальную прогонку, но оставляем связи на границах
    # Мы выражаем x_i = alpha_i * x_{i+1} + beta_i для всех i в блоке (включая последний!)
    alpha = np.empty(local_n, dtype=np.float64)
    beta = np.empty(local_n, dtype=np.float64)

    # Начинаем с первой строки блока
    if local_start == 0:
        # первая строка глобальной системы
        alpha[0] = c_local[0] / b[0]
        beta[0] = d[0] / b[0]
    else:
        # строка зависит от предыдущего блока → пока не может быть завершена
        # но мы всё равно делаем формальную прогонку, чтобы получить alpha/beta на выходе
        alpha[0] = c_local[0] / b[0]
        beta[0] = d[0] / b[0]

    for i in range(1, local_n):
        denom = b[i] - a_local[i] * alpha[i-1]
        alpha[i] = c_local[i] / denom
        beta[i] = (d[i] - a_local[i] * beta[i-1]) / denom

    # Теперь alpha[-1], beta[-1] связывают последнюю переменную блока с "внешней" x_{next}
    # Этап 2.3: Обмен и сбор редуцированной системы
    # Каждый процесс предоставляет:

    # Собираем на root граничные коэффициенты
    all_beta_last = comm.gather(beta[-1], root=0)
    all_alpha_last = comm.gather(alpha[-1], root=0)
    all_local_n = comm.gather(local_n, root=0)
    all_starts = comm.gather(local_start, root=0)

    if rank == 0:
        # Строим редуцированную систему размера size
        A_red = np.zeros(size, dtype=np.float64)  # поддиагональ редуц. системы
        B_red = np.zeros(size, dtype=np.float64)  # диагональ
        C_red = np.zeros(size, dtype=np.float64)  # наддиагональ
        D_red = np.zeros(size, dtype=np.float64)  # правая часть

        for i in range(size):
            n_i = all_local_n[i]
            start_i = all_starts[i]
            end_i = start_i + n_i

            alpha_last = all_alpha_last[i]
            beta_last = all_beta_last[i]

            if i == 0:
                # Первый блок: первое уравнение — глобальное x0
                B_red[0] = 1.0
                C_red[0] = -alpha_last if n_i > 1 or size > 1 else 0.0
                D_red[0] = beta_last
            elif i == size - 1:
                # Последний блок: последнее уравнение — глобальное x_{N-1}
                A_red[i] = 1.0
                B_red[i] = 0.0
                D_red[i] = beta_last
                B_red[i] = 1.0
                A_red[i] = 0.0
                D_red[i] = beta_last
            else:
                # Внутренний блок: связь между блоками
                A_red[i] = 1.0
                B_red[i] = -all_alpha_last[i-1]  # от предыдущего блока
                C_red[i] = -alpha_last
                D_red[i] = beta_last - all_beta_last[i-1] * 0  # приближение

        # редуцированная система строится из уравнений, связывающих последние и первые переменные соседей.

        # Переопределим:
        red_a = []
        red_b = []
        red_c = []
        red_d = []

        for i in range(size):
            if i == 0:
                # Первый блок: уравнение: x0 = alpha0 * x1 + beta0
                # Но x0 — это X0, x1 — либо внутри блока, либо X1
                # Предположим, что каждый блок даёт связь: X_i = P_i * X_{i+1} + Q_i
                P = all_alpha_last[i]
                Q = all_beta_last[i]
                red_b.append(1.0)
                red_c.append(-P)
                red_d.append(Q)
                if size == 1:
                    red_b[0] = 1.0
                    red_c[0] = 0.0
                    red_d[0] = Q
            elif i == size - 1:
                # Последний блок: последнее уравнение: x_last = beta_last (т.к. c=0)
                # Но x_last = X_i (если блок один), или выражено через X_i
                # В простейшем случае: X_i = beta_last
                red_a.append(0.0)
                red_b.append(1.0)
                red_c.append(0.0)
                red_d.append(all_beta_last[i])
            else:
                P = all_alpha_last[i]
                Q = all_beta_last[i]
                red_a.append(1.0)
                red_b.append(0.0)  # placeholder
                red_c.append(-P)
                red_d.append(Q)

        x_global = thomas_solve(a_global, b_global, c_global, d_global)
        all_x = x_global
    else:
        all_x = None

    # Распределение результата
    x_local = np.empty(local_n, dtype=np.float64)
    comm.Scatterv([all_x, counts, displs, MPI.DOUBLE], x_local, root=0)

    # Вывод (опционально)
    # Собираем всё для проверки
    x_gathered = np.empty(N, dtype=np.float64)
    comm.Gatherv(x_local, [x_gathered, counts, displs, MPI.DOUBLE], root=0)

    if rank == 0:
        residual = np.abs(
            (np.concatenate(([0], x_gathered[:-1])) * (-1) +
             x_gathered * 2 +
             np.concatenate((x_gathered[1:], [0])) * (-1)) - d_global
        )
        print("Максимальная невязка:", np.max(residual))

if __name__ == "__main__":
    main()
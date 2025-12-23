import numpy as np
import time

# Последовательный метод прогонки (Thomas algorithm)
def thomas(a, b, c, d):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    n = len(b)

    if n == 1:
        return np.array([d[0] / b[0]])

    cp = np.zeros(n - 1)
    dp = np.zeros(n)

    # прямой ход
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / \
                (b[n - 1] - a[n - 2] * cp[n - 2])

    # обратный ход
    x = np.zeros(n)
    x[-1] = dp[-1]

    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


# Генерация трехдиагональной системы (диагональное преобладание)

def generate_system(n, seed=0):
    rng = np.random.default_rng(seed)

    a = rng.uniform(-1.0, 1.0, size=n - 1)
    c = rng.uniform(-1.0, 1.0, size=n - 1)
    b = rng.uniform(2.0, 3.0, size=n)

    # диагональное преобладание
    b[1:-1] += np.abs(a[:-1]) + np.abs(c[1:])
    b[0] += np.abs(c[0])
    b[-1] += np.abs(a[-1])

    d = rng.uniform(-1.0, 1.0, size=n)

    return a, b, c, d


# Тест корректности

def test_correctness():
    print("=== Проверка корректности ===")
    n = 5
    a = -1.0 * np.ones(n - 1)
    b = 4.0 * np.ones(n)
    c = -1.0 * np.ones(n - 1)
    d = np.ones(n)

    x = thomas(a, b, c, d)

    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = b
    A[np.arange(n - 1) + 1, np.arange(n - 1)] = a
    A[np.arange(n - 1), np.arange(n - 1) + 1] = c

    residual = np.linalg.norm(A @ x - d)
    print("x =", x)
    print("||Ax - d|| =", residual)


# Замеры времени и сравнение с NumPy

def benchmark():
    print("\n=== Замеры времени ===")
    for n in [100, 1000, 10000]:
        a, b, c, d = generate_system(n)

        t0 = time.perf_counter()
        x = thomas(a, b, c, d)
        t1 = time.perf_counter()

        A = np.zeros((n, n))
        A[np.arange(n), np.arange(n)] = b
        A[np.arange(n - 1) + 1, np.arange(n - 1)] = a
        A[np.arange(n - 1), np.arange(n - 1) + 1] = c

        t2 = time.perf_counter()
        x_np = np.linalg.solve(A, d)
        t3 = time.perf_counter()

        err = np.linalg.norm(x - x_np) / np.linalg.norm(x_np)

        print(f"n = {n:6d} | "
              f"Thomas: {t1 - t0:.6f} s | "
              f"NumPy: {t3 - t2:.6f} s | "
              f"rel error: {err:.2e}")


# Точка входа

if __name__ == "__main__":
    test_correctness()
    benchmark()
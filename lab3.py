import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ============================================================
# 1. ПІДГОТОВКА ВХІДНИХ ДАНИХ
# ============================================================
def prepare_csv(filename="data_lab3.csv"):
    """Формує CSV-файл із середньомісячними температурами."""
    data = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
        (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
        (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
        (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Month", "Temp"])
        writer.writerows(data)
    print(f"[INFO] Дані записано у файл {filename}")


def read_csv(filename="data_lab3.csv"):
    """Зчитує місяці та температури з CSV-файлу."""
    x, y = [], []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # пропустити заголовок
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)


# ============================================================
# 2. ФУНКЦІЇ МНК
# ============================================================
def form_matrix(x, m):
    """Формує матрицю B розміром (m+1, m+1)."""
    B = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(x ** (k + l))
    return B


def form_vector(x, y, m):
    """Формує вектор C розміром (m+1)."""
    C = np.zeros(m + 1)
    for k in range(m + 1):
        C[k] = np.sum(y * x ** k)
    return C


def gauss_solve(A, b):
    """Метод Гауса з вибором головного елемента по стовпцях."""
    n = len(b)
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    # Прямий хід
    for k in range(n - 1):
        max_row = k + int(np.argmax(np.abs(A[k:, k])))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol


def polynomial(x, coef):
    """Обчислює значення многочлена за коефіцієнтами."""
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * x ** i
    return y_poly


def variance(y_true, y_approx):
    """Обчислює дисперсію."""
    return float(np.sqrt(np.mean((y_true - y_approx) ** 2)))


# ============================================================
# 3. ВИБІР ОПТИМАЛЬНОГО СТУПЕНЯ ПОЛІНОМА
# ============================================================
def main():
    prepare_csv()
    x, y = read_csv()

    max_degree = 10
    variances = []
    coefficients = []

    print("\n=== Обчислення дисперсії для m = 1..10 ===")
    for m in range(1, max_degree + 1):
        B = form_matrix(x, m)
        C = form_vector(x, y, m)
        coef = gauss_solve(B, C)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)
        coefficients.append(coef)
        print(f"m = {m:2d} | дисперсія δ = {var:.6f}")

    optimal_m = int(np.argmin(variances)) + 1
    print(f"\n[INFO] Оптимальний степінь полінома: m = {optimal_m}")
    print(f"[INFO] Мінімальна дисперсія: {min(variances):.6f}")

    # ========================================================
    # 4. ПОБУДОВА АПРОКСИМАЦІЇ З ОПТИМАЛЬНИМ СТЕПЕНЕМ
    # ========================================================
    B = form_matrix(x, optimal_m)
    C = form_vector(x, y, optimal_m)
    coef_opt = gauss_solve(B, C)
    y_approx = polynomial(x, coef_opt)

    print(f"\n[INFO] Коефіцієнти полінома (m = {optimal_m}):")
    for i, c in enumerate(coef_opt):
        print(f"  a[{i}] = {c:.8f}")

    # ========================================================
    # 5. ПРОГНОЗ НА НАСТУПНІ 3 МІСЯЦІ
    # ========================================================
    x_future = np.array([25.0, 26.0, 27.0])
    y_future = polynomial(x_future, coef_opt)
    print("\n[INFO] Прогноз температури на місяці 25–27:")
    for month, temp in zip(x_future, y_future):
        print(f"  Місяць {int(month)}: {temp:.4f} °C")

    # ========================================================
    # 6. ПОХИБКА АПРОКСИМАЦІЇ
    # ========================================================
    error = np.abs(y - y_approx)
    print(f"\n[INFO] Максимальна похибка: {np.max(error):.6f}")
    print(f"[INFO] Середня похибка:    {np.mean(error):.6f}")

    # ========================================================
    # 7. ВІЗУАЛІЗАЦІЯ
    # ========================================================
    # ---- Графік 1: Дисперсія vs степінь ----
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, max_degree + 1), variances, "o-",
             color="navy", linewidth=2, markersize=7)
    plt.axvline(optimal_m, color="red", linestyle="--",
                label=f"Оптимум m = {optimal_m}")
    plt.xlabel("Степінь полінома m", fontsize=12)
    plt.ylabel("Дисперсія δ", fontsize=12)
    plt.title("Графік 1: Залежність дисперсії від степеня полінома", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("graph1_variance.png", dpi=150)

    # ---- Графік 2: Апроксимація + фактичні дані + прогноз ----
    x_smooth = np.linspace(x.min(), 27, 400)
    y_smooth = polynomial(x_smooth, coef_opt)

    plt.figure(figsize=(10, 5.5))
    plt.plot(x_smooth, y_smooth, "-", color="navy",
             linewidth=2, label=f"Апроксимація (m = {optimal_m})")
    plt.scatter(x, y, color="red", s=60, zorder=5,
                label="Експериментальні дані")
    plt.scatter(x_future, y_future, color="green", marker="*",
                s=220, zorder=6, label="Прогноз (місяці 25–27)")
    plt.xlabel("Місяць", fontsize=12)
    plt.ylabel("Температура, °C", fontsize=12)
    plt.title("Графік 2: Апроксимація МНК та прогноз температури", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("graph2_approximation.png", dpi=150)

    # ---- Графік 3: Похибка ----
    plt.figure(figsize=(9, 5))
    plt.plot(x, error, "o-", color="darkred", linewidth=2, markersize=7)
    plt.xlabel("Місяць", fontsize=12)
    plt.ylabel("Похибка ε(x) = |f(x) − φ(x)|", fontsize=12)
    plt.title("Графік 3: Похибка апроксимації", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("graph3_error.png", dpi=150)

    # ---- Графік 4: Похибка для різних m ----
    plt.figure(figsize=(10, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 1, max_degree))
    for idx, m in enumerate(range(1, max_degree + 1)):
        err_nodes = np.abs(y - polynomial(x, coefficients[idx]))
        plt.plot(x, err_nodes, "o-", color=colors[idx],
                 label=f"m = {m}", alpha=0.75, markersize=4)
    plt.xlabel("Місяць", fontsize=12)
    plt.ylabel("Похибка ε(x)", fontsize=12)
    plt.title("Графік 4: Похибка для різних степенів m = 1..10", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("graph4_errors_m.png", dpi=150)

    plt.show()
    print("\n[INFO] Усі графіки збережено у поточній директорії.")


if __name__ == "__main__":
    main()
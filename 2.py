""" Скрипт ... .

    Предполагает нахождение файла "NdYAG temperature.txt" в той же папке, что и скрипт. 
    Требует следующей настройки COMSOL-модели:
    - Активный элемент расположен вдоль оси 'x' (с началом в x=0).
    - Выравнивание активного элемента по центру (ось в y=0, z=0).
    - Рассчитаны значения температуры активного элемента.
    - Экспорт данных без принудительной Regular Grid, без Header, с разделителем-запятой.
"""

import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

# Получение полного пути к файлу. Должен лежать в той же папке, что и скрипт.
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = "NdYAG temperature.txt" # имя файла с данными
full_path = os.path.join(script_dir, filename)

# Загрузка данных
data = np.loadtxt(full_path, delimiter=",", dtype=float)
if data.shape[1] != 4:
    raise ValueError("Ожидался файл с 4 столбцами: x,y,z,T")
if not np.isfinite(data).all():
    raise ValueError("В данных есть NaN или Inf")

# Распаковка столбцов
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
T = data[:, 3]

# Интерполяция температуры (триангуляция Делоне)
points = np.column_stack((x, y, z))         # создание массива точек (x,y,z)
interp_T = LinearNDInterpolator(points, T)

# Создание равномерной сетки вдоль оси активного элемента
x_min, x_max = x.min(), x.max()
Nx = 300                                # кол-во точек; можно 200–500
x_line = np.linspace(x_min, x_max, Nx)

# Определение реального радиуса
r_nodes = np.sqrt(y**2 + z**2)      # массив радиусов узловых точек
perc = 98                           # процентиль; для более грубых сеток 95-98
R = np.percentile(r_nodes, perc)    # исключение выбросов за реальный радиус

# Фит-радиус (область квадратичного профиля температуры)
R_fit = 0.3 * R
# Добавить автоматическое вычисление R_fit!

# Создание равномерной сетки в квадрате R_fit
Ny = Nz = 70                                # кол-во точек по 'y' и 'z'; можно 40–100
y_grid = np.linspace(-R_fit, R_fit, Ny)
z_grid = np.linspace(-R_fit, R_fit, Nz)

# Отбрасывание точек, выходящих за круг радиуса R_fit
YZ = [(yy, zz) for yy in y_grid for zz in z_grid if yy**2 + zz**2 <= R_fit**2]

# Вычисление T_ref как среднего значения T вблизи оси (где r < 0.05*R)
axis_mask = r_nodes < 0.05 * R
T_ref = np.mean(T[axis_mask])

# Пустые контейнеры для результатов
Theta = []
Y = []
Z = []

# Вычисление Θ(y,z) для каждой точки (y,z) в YZ
for yy, zz in YZ:
    # Интерполяция T вдоль линии x для данной (y,z)
    T_line = interp_T(x_line, yy, zz)

    # защита от выхода за область интерполяции
    if np.any(np.isnan(T_line)):
        print(f"Warning: Interpolation failed at (y={yy:.2f}, z={zz:.2f}). Skipping this point.")
        continue

    # Интегрирование по x для получения Θ(y,z)
    theta = np.trapezoid(T_line - T_ref, x_line)

    # Сохранение результатов
    Y.append(yy)
    Z.append(zz)
    Theta.append(theta)

# Конвертация в numpy-массивы
Y = np.array(Y)
Z = np.array(Z)
Theta = np.array(Theta)

# Вывод результатов
# print("Computed Θ(y,z) for", Theta.size, "rays")
# print("Θ range:", Theta.min(), Theta.max())
# plt.tricontourf(Y, Z, Theta, levels=30)
# plt.gca().set_aspect('equal')
# plt.colorbar(label=r'$\Theta(y,z)$')
#plt.show()

# дизайн-матрица
M = np.column_stack([
    Y**2,
    Z**2,
    Y*Z,
    np.ones_like(Y)
])

# МНК
coeffs, *_ = np.linalg.lstsq(M, Theta, rcond=None)

A, B, C, D = coeffs


# i0 = np.argmax(Theta)
# y0, z0 = Y[i0], Z[i0]
# Yp = Y - y0
# Zp = Z - z0

# M = np.column_stack([
#     Yp**2, Zp**2,
#     Yp, Zp,
#     np.ones_like(Yp)
# ])
# A, B, Ey, Ez, D = np.linalg.lstsq(M, Theta, rcond=None)[0]

lambda0 = 1064e-9      # длина волны, м
dn_dT   = 7.3e-6      # К⁻¹ (пример!)
k = 2*np.pi / lambda0

print("Fitted coefficients: A=%.2e, B=%.2e, C=%.2e, D=%.2e" % (A, B, C, D))

f_y = 1 / (2 * A * dn_dT)
f_z = 1 / (2 * B * dn_dT)

print("Focal lengths: f_y = %.10f m, f_z = %.10f m" % (f_y, f_z))

Theta_fit = M @ coeffs
residual = Theta - Theta_fit

print(np.std(residual) / np.std(Theta) * 100, "% residual error")

print(Theta.min(), Theta.max())

mask = np.abs(Z) < 0.02 * R
Yc = Y[mask]
Thetac = Theta[mask]

# сортировка
idx = np.argsort(Yc)
Yc = Yc[idx]
Thetac = Thetac[idx]

plt.plot(Yc, Thetac, 'o-')
plt.xlabel('y')
plt.ylabel('Theta')
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class RealisticSatelliteSystem:
    def __init__(self, relativistic=True):
        """
        Реалистичная система для моделирования движения спутников вокруг Земли
        с учетом или без учета релятивистских эффектов (ОТО)
        """
        # Константы
        self.G = 6.67430e-11  # м³/(кг·с²)
        self.c = 299792458  # м/с
        self.c2 = self.c ** 2

        # Параметры Земли (Stjarnhimlen(сайт с которого брался метод расчета))
        self.M_earth = 5.9742e24  # кг
        self.R_earth = 6378137.0  # м
        self.J2 = 1.08263e-3  # Коэффициент сжатия Земли
        self.omega_earth = 7.292115e-5  # рад/с (угловая скорость вращения)

        # Гравитационный параметр Земли
        self.mu = self.G * self.M_earth

        # Флаги для включения эффектов(не трогать)
        self.relativistic = relativistic
        self.include_J2 = True

        self.results = {}

    def schwarzschild_radius(self):
        """Радиус Шварцшильда Земли"""
        return 2 * self.G * self.M_earth / self.c2

    def geodesic_equation(self, t, state):
        """
        Уравнения геодезической в постньютоновском приближении
        """
        x, y, z, vx, vy, vz = state

        r_vec = np.array([x, y, z])
        r = np.linalg.norm(r_vec)
        v_vec = np.array([vx, vy, vz])
        v2 = np.dot(v_vec, v_vec)

        # Базовое ньютоновское ускорение
        a_newton = -self.mu * r_vec / r ** 3

        if self.relativistic:
            # Релятивистская поправка к ускорению (1PN)
            term1 = (4 * self.mu / r - v2) * r_vec
            term2 = 4 * np.dot(r_vec, v_vec) * v_vec
            a_rel = (self.mu / (self.c2 * r ** 3)) * (term1 + term2)
            a_total = a_newton + a_rel
        else:
            a_total = a_newton

        if self.include_J2 and r > self.R_earth:
            # Влияние сжатия Земли (J2-эфффект)
            x, y, z = r_vec
            r2 = r ** 2
            r5 = r ** 5

            # J2
            j2_factor = 1.5 * self.J2 * self.mu * self.R_earth ** 2 / r5

            a_j2_x = j2_factor * x * (5 * z ** 2 / r2 - 1)
            a_j2_y = j2_factor * y * (5 * z ** 2 / r2 - 1)
            a_j2_z = j2_factor * z * (5 * z ** 2 / r2 - 3)

            a_total += np.array([a_j2_x, a_j2_y, a_j2_z])

        return [vx, vy, vz, a_total[0], a_total[1], a_total[2]]

    def newtonian_equation(self, t, state):
        """
        Чисто ньютоновские уравнения движения
        """
        x, y, z, vx, vy, vz = state

        r_vec = np.array([x, y, z])
        r = np.linalg.norm(r_vec)

        # Ньютоновское ускорение
        a_newton = -self.mu * r_vec / r ** 3

        if self.include_J2 and r > self.R_earth:
            # Влияние сжатия Земли
            x, y, z = r_vec
            r2 = r ** 2
            r5 = r ** 5

            j2_factor = 1.5 * self.J2 * self.mu * self.R_earth ** 2 / r5

            a_j2_x = j2_factor * x * (5 * z ** 2 / r2 - 1)
            a_j2_y = j2_factor * y * (5 * z ** 2 / r2 - 1)
            a_j2_z = j2_factor * z * (5 * z ** 2 / r2 - 3)

            a_newton += np.array([a_j2_x, a_j2_y, a_j2_z])

        return [vx, vy, vz, a_newton[0], a_newton[1], a_newton[2]]

    def simulate_satellite(self, initial_state, t_span, t_eval, name="Satellite"):
        """
        Моделирование движения спутника
        """
        print(f"  Моделирование {name}...")

        # Ньютоновское моделирование
        sol_newton = solve_ivp(
            self.newtonian_equation,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )

        # Релятивистское моделирование
        sol_rel = solve_ivp(
            self.geodesic_equation,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )

        # Расчет орбитальных параметров
        def calculate_orbital_parameters(t, y):
            """Расчет орбитальных параметров из состояния"""
            positions = y[:3, :]
            velocities = y[3:, :]

            n_points = positions.shape[1]
            params = {
                'r': np.zeros(n_points),
                'v': np.zeros(n_points),
                'angular_momentum': np.zeros((3, n_points)),
            }

            for i in range(n_points):
                r_vec = positions[:, i]
                v_vec = velocities[:, i]

                r = np.linalg.norm(r_vec)
                v = np.linalg.norm(v_vec)

                # Удельный момент импульса
                h_vec = np.cross(r_vec, v_vec)

                params['r'][i] = r
                params['v'][i] = v
                params['angular_momentum'][:, i] = h_vec

            return params

        newton_params = calculate_orbital_parameters(sol_newton.t, sol_newton.y)
        rel_params = calculate_orbital_parameters(sol_rel.t, sol_rel.y)

        self.results[name] = {
            'newton': {'t': sol_newton.t, 'y': sol_newton.y, 'params': newton_params},
            'relativistic': {'t': sol_rel.t, 'y': sol_rel.y, 'params': rel_params}
        }

        return self.results[name]

    def create_satellites(self):
        """Создание различных спутников с акцентом на GPS"""
        satellites = []

        # 1. GPS спутник (реальная орбита GPS)
        r_gps = 26560e3  # 20180 км над поверхностью
        v_gps = np.sqrt(self.mu / r_gps)
        # Наклонение 55° для GPS
        inclination_gps = np.radians(55)
        initial_gps = [
            r_gps, 0, 0,
            0, v_gps * np.cos(inclination_gps), v_gps * np.sin(inclination_gps)
        ]
        satellites.append({
            'name': 'GPS',
            'state': initial_gps,
            'color': 'blue'
        })

        # 2. ГЛОНАСС (орбита отличается от GPS)
        r_glonass = 25500e3  # 19122 км над поверхностью
        v_glonass = np.sqrt(self.mu / r_glonass)
        inclination_glonass = np.radians(64.8)
        initial_glonass = [
            r_glonass, 0, 0,
            0, v_glonass * np.cos(inclination_glonass), v_glonass * np.sin(inclination_glonass)
        ]
        satellites.append({
            'name': 'ГЛОНАСС',
            'state': initial_glonass,
            'color': 'green'
        })

        # 3. Galileo (европейская система, орбита выше)
        r_galileo = 29600e3  # 23222 км над поверхностью
        v_galileo = np.sqrt(self.mu / r_galileo)
        inclination_galileo = np.radians(56)
        initial_galileo = [
            r_galileo, 0, 0,
            0, v_galileo * np.cos(inclination_galileo), v_galileo * np.sin(inclination_galileo)
        ]
        satellites.append({
            'name': 'Galileo',
            'state': initial_galileo,
            'color': 'orange'
        })

        # 4. Низкая околоземная орбита (МКС) - для прикола
        h = 400e3  # 400 км
        r = self.R_earth + h
        v_circ = np.sqrt(self.mu / r)

        initial_iss = [r, 0, 0, 0, v_circ, 0]
        satellites.append({
            'name': 'Низкая орбита (МКС)',
            'state': initial_iss,
            'color': 'red'
        })

        return satellites


def analyze_results(system, results):
    """
    Анализ и визуализация результатов с акцентом на GPS
    """

    # Создаем графики
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'GPS': 'blue', 'ГЛОНАСС': 'green', 'Galileo': 'orange', 'Низкая орбита (МКС)': 'red'}

    # 1. Накопленная ошибка в положении из-за ОТО
    ax1.set_title('Накопленная ошибка в положении из-за ОТО')
    ax1.set_xlabel('Время (дни)')
    ax1.set_ylabel('Ошибка положения (метры)')
    ax1.grid(True, alpha=0.3)

    # 2. Накопленная ошибка времени
    ax2.set_title('Накопленная ошибка времени из-за ОТО')
    ax2.set_xlabel('Время (дни)')
    ax2.set_ylabel('Ошибка времени (микросекунды)')
    ax2.grid(True, alpha=0.3)

    # 3. Ошибка в определении положения за день
    ax3.set_title('Ошибка в позиционировании за сутки (Теоретическая)')
    ax3.set_xlabel('Спутниковая система')
    ax3.set_ylabel('Ошибка (км/день)')
    ax3.grid(True, alpha=0.3)

    # Для вычисления суточной ошибки
    daily_position_errors = {}
    daily_time_errors = {}

    for sat_name, result in results.items():
        color = colors[sat_name]

        # Данные
        t_newton = result['newton']['t']
        y_newton = result['newton']['y']
        y_rel = result['relativistic']['y']

        # Время в дни
        t_days = t_newton / 86400

        # 1. Накопленная ошибка в положении (разница траекторий)
        delta_r = np.sqrt(np.sum((y_rel[:3, :] - y_newton[:3, :]) ** 2, axis=0))
        ax1.plot(t_days, delta_r, color=color, linewidth=2, label=f'{sat_name}')

        # 2. Накопленная ошибка времени
        if 'GPS' in sat_name or 'ГЛОНАСС' in sat_name or 'Galileo' in sat_name:
            # Получаем данные из релятивистской модели
            positions = y_rel[:3, :]  # x, y, z
            velocities = y_rel[3:, :]  # vx, vy, vz

            n_points = positions.shape[1]
            time_error = np.zeros(n_points)

            # Параметры для расчета
            mu = system.mu
            c = system.c
            c2 = c ** 2
            R_earth = system.R_earth

            # Рассчитываем накопленную ошибку времени
            for i in range(1, n_points):
                # Расстояние от центра Земли
                r_vec = positions[:, i]
                r = np.linalg.norm(r_vec)

                # Скорость спутника
                v_vec = velocities[:, i]
                v = np.linalg.norm(v_vec)
                v2 = v ** 2

                # Гравитационный потенциал на поверхности Земли (для наземных часов)
                U_surface = mu / R_earth
                # и на высоте спутника
                U_satellite = mu / r

                # Мгновенное относительное замедление времени
                # Формула: dt/t = (ΔU)/c² - v²/(2c²)
                dt_instant = (U_surface - U_satellite) / c2 - v2 / (2 * c2)

                # Шаг времени
                dt = t_newton[i] - t_newton[i - 1]

                # Накопленная ошибка времени (интегрируем)
                time_error[i] = time_error[i - 1] + dt_instant * dt

            # Конвертируем в микросекунды
            time_error_microsec = time_error * 1e6

            ax2.plot(t_days, time_error_microsec, color=color, linewidth=2, label=f'{sat_name}')

            # Вычисляем суточную ошибку времени
            one_day_idx = np.argmax(t_newton >= 86400)
            if one_day_idx > 0 and one_day_idx < len(time_error_microsec):
                daily_time_errors[sat_name] = time_error_microsec[one_day_idx]

        # Вычисляем суточную ошибку положения (только для вывода в журнал)
        one_day_idx = np.argmax(t_newton >= 86400)
        if one_day_idx > 0:
            daily_position_error = delta_r[one_day_idx]
            daily_position_errors[sat_name] = daily_position_error

    # 3. Ошибка в определении положения за день (на основе теоретического расчета времени)
    if daily_time_errors:
        sat_names = list(daily_time_errors.keys())
        # Конвертируем ошибку времени в ошибку положения (км)
        error_values_km = [err * system.c / 1e9 for err in daily_time_errors.values()]
        bar_colors = [colors[name] for name in sat_names]

        bars = ax3.bar(range(len(sat_names)), error_values_km,
                       color=bar_colors, alpha=0.8)
        ax3.set_xticks(range(len(sat_names)))
        ax3.set_xticklabels(sat_names, rotation=45, ha='right')

        # Добавляем значения на столбцы
        for bar, value in zip(bars, error_values_km):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height * 1.02,
                     f'{value:.2f} км', ha='center', va='bottom', fontsize=9)

    # Настраиваем легенды
    ax1.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')

    # Добавляем текстовую информацию
    info_text = f"""
    Параметры моделирования:
    • Масса Земли: {system.M_earth:.3e} кг
    • Радиус Земли: {system.R_earth / 1000:.0f} км
    • Гравитационный параметр (μ): {system.mu:.3e} м³/с²
    • Включены эффекты: {'ОТО' if system.relativistic else 'только Ньютон'} + J2
    • Длительность: {t_days[-1]:.0f} дней
    """

    plt.figtext(0.02, 0.02, info_text, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle('Лабораторная работа: Влияние ОТО на точность спутниковых навигационных систем',
                 fontsize=12, fontweight='bold')

    return fig, daily_position_errors


def demonstrate_gps_effects():
    """
    Демонстрация эффектов для GPS систем (Теоретический расчет)
    """

    # Параметры
    G = 6.67430e-11
    c = 299792458
    c2 = c ** 2
    M_earth = 5.9742e24
    R_earth = 6378137.0

    # Данные для разных спутников
    systems = [
        {'name': 'GPS', 'r': 26560e3, 'inclination': 55},
        {'name': 'ГЛОНАСС', 'r': 25500e3, 'inclination': 64.8},
        {'name': 'Galileo', 'r': 29600e3, 'inclination': 56},
        {'name': 'Низкая орбита (МКС)', 'r': R_earth + 400e3, 'inclination': 51.6},
    ]

    results = []

    for system_data in systems:
        r = system_data['r']
        v = np.sqrt(G * M_earth / r)  # м/с

        # Расчёт релятивистских эффектов
        mu = G * M_earth
        delta_grav = mu / c2 * (1 / R_earth - 1 / r)  # гравитационный эффект (ОТО)
        delta_kin = -0.5 * (v / c) ** 2  # кинематический эффект (СТО)
        delta_total = delta_grav + delta_kin  # суммарный эффект

        # Ошибка времени за день
        seconds_per_day = 86400
        time_error = delta_total * seconds_per_day * 1e6  # микросекунды в день

        # Ошибка позиционирования за день
        positioning_error = delta_total * seconds_per_day * c  # метры в день

        results.append({
            'name': system_data['name'],
            'height': (r - R_earth) / 1000,
            'velocity': v / 1000,
            'delta_grav': delta_grav,
            'delta_kin': delta_kin,
            'delta_total': delta_total,
            'time_error': time_error,
            'positioning_error': positioning_error
        })

    # Вывод результатов
    print(f"\nСравнение теоретически рассчитанных релятивистских эффектов (1 день):")
    print(f"{'Система':<20} {'Высота (км)':<12} {'Скорость (км/с)':<15} {'Δt (мкс/день)':<15} {'Δx (км/день)':<15}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['name']:<20} {r['height']:<12.0f} {r['velocity']:<15.2f} {r['time_error']:<15.2f} {r['positioning_error'] / 1000:<15.2f}")

    print("\n" + "=" * 80)
    print("Ключевые выводы для GPS (Теоретический расчет):")
    print("=" * 80)

    gps_result = [r for r in results if r['name'] == 'GPS'][0]

    print(f"\n1. Релятивистские эффекты GPS (в нс/сек):")
    print(f"    Гравитационный эффект (ОТО): +{gps_result['delta_grav'] * 1e9:.2f} нс/сек")
    print(f"    Кинематический эффект (СТО): {gps_result['delta_kin'] * 1e9:.2f} нс/сек")
    print(f"    Суммарный эффект (спешка): +{gps_result['delta_total'] * 1e9:.2f} нс/сек")

    print(f"\n2. Накопление за сутки:")
    print(f"    Ошибка времени: {gps_result['time_error']:.2f} микросекунд")
    print(f"    Ошибка позиционирования: {gps_result['positioning_error'] / 1000:.2f} километров")


def main(simulation_days=30):

    # Создаем систему с учетом ОТО
    relativistic_system = RealisticSatelliteSystem(relativistic=True)

    # Создаем спутники
    satellites = relativistic_system.create_satellites()

    # Параметры моделирования
    t_span = (0, simulation_days * 86400)  # секунд
    t_eval = np.linspace(t_span[0], t_span[1], 1500)  # точек для расчета(1500 приемлимая точность)

    print(f"Параметры моделирования:")
    print(f"  Длительность: {simulation_days} дней")
    print(f"  Включен эффект J2: {relativistic_system.include_J2}")
    print(f"  Включены релятивистские эффекты: {relativistic_system.relativistic}")
    print(f"\nМоделирование спутниковых систем:")

    # Моделируем каждый спутник
    all_results = {}
    for sat in satellites:
        result = relativistic_system.simulate_satellite(
            sat['state'],
            t_span,
            t_eval,
            name=sat['name']
        )
        all_results[sat['name']] = result

    # Анализируем и визуализируем результаты
    fig, daily_pos_errors = analyze_results(relativistic_system, all_results)

    # Демонстрация теоретических эффектов GPS
    demonstrate_gps_effects()

    # Подробный анализ для навигационных систем
    print("\n" + "=" * 80)
    print("Результаты моделирования (Накопленная ошибка положения за 1 день):")
    print("=" * 80)

    for sat_name, error in daily_pos_errors.items():
        # Ошибка положения (метры)
        if error > 100:
             print(f"  {sat_name}: {error:.2f} м (Значительное расхождение траекторий!)")
        elif error > 1:
             print(f"  {sat_name}: {error:.2f} м (Небольшое, но заметное расхождение)")
        else:
             print(f"  {sat_name}: {error:.3f} м (Минимальное расхождение)")


    return fig
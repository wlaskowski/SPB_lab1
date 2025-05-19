import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess

# automatyczna instalacja bibliotek (jeśli brakuje)
def ensure_packages_installed(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_packages_installed(["numpy", "matplotlib"])

# definicja wszystkich scenariuszy modelu w słowniku
BIO_SCENARIOS = {
    "A": {"params": {"p1": 8.8, "p2": 440, "p3": 100, "d1": 1.375e-14, "d2": 1.375e-4, "d3": 3e-5, "k1": 1.925e-4, "k2": 1e5, "k3": 1.5e5},
           "conditions": {"dna_damage": False, "siRNA": False, "pten_active": True},
           "initial": [30, 50, 50, 80]},
    "B": {"params": {"p1": 8.8, "p2": 440, "p3": 100, "d1": 1.375e-14, "d2": 1.375e-4, "d3": 3e-5, "k1": 1.925e-4, "k2": 1e5, "k3": 1.5e5},
           "conditions": {"dna_damage": True, "siRNA": False, "pten_active": True},
           "initial": [26800, 154200, 11080, 15900]},
    "C": {"params": {"p1": 8.8, "p2": 440, "p3": 100, "d1": 1.375e-14, "d2": 1.375e-4, "d3": 3e-5, "k1": 1.925e-4, "k2": 1e5, "k3": 1.5e5},
           "conditions": {"dna_damage": True, "siRNA": False, "pten_active": False},
           "initial": [26800, 154200, 11080, 15900]},
    "D": {"params": {"p1": 8.8, "p2": 440, "p3": 100, "d1": 1.375e-14, "d2": 1.375e-4, "d3": 3e-5, "k1": 1.925e-4, "k2": 1e5, "k3": 1.5e5},
           "conditions": {"dna_damage": True, "siRNA": True, "pten_active": False},
           "initial": [26800, 154200, 11080, 15900]}
}

# funkcja rk4
def simulate(system_eq, y_start, t_span, args):
    y_result = np.zeros((len(t_span), len(y_start)))
    y_result[0] = y_start
    for i in range(1, len(t_span)):
        dt = t_span[i] - t_span[i-1]
        y = y_result[i-1]
        k1 = system_eq(t_span[i-1], y, *args)
        k2 = system_eq(t_span[i-1] + dt/2, y + dt/2*k1, *args)
        k3 = system_eq(t_span[i-1] + dt/2, y + dt/2*k2, *args)
        k4 = system_eq(t_span[i-1] + dt, y + dt*k3, *args)
        y_result[i] = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_result

# regulacja parametrów
def biological_dynamics(t, y, p, cond):
    p53, mdm2_n, mdm2_c, pten = y
    p_mod = p.copy()

# modyfikacje
    if cond["siRNA"]:
        p_mod["p2"] *= 0.02
    if cond["pten_active"] == False:
        p_mod["p3"] = 0
    if not cond["dna_damage"]:
        p_mod["d2"] *= 0.1

# równania różniczkowe
    synthesis = p_mod["p1"]
    feedback = p_mod["d1"] * p53 * (mdm2_n ** 2)
    mdm2_prod = p_mod["p2"] * p53**4 / (p53**4 + p_mod["k2"]**4)
    pten_reg = p_mod["k3"]**2 / (p_mod["k3"]**2 + pten**2)

    dp53 = synthesis - feedback
    dmdm2_c = mdm2_prod - p_mod["k1"] * pten_reg * mdm2_c - p_mod["d2"] * mdm2_c
    dmdm2_n = p_mod["k1"] * pten_reg * mdm2_c - p_mod["d2"] * mdm2_n
    dpten = p_mod["p3"] * p53**4 / (p53**4 + p_mod["k2"]**4) - p_mod["d3"] * pten

    return np.array([dp53, dmdm2_n, dmdm2_c, dpten])

# tworzenie wykresów
def generate_plot(time, data, title):
    labels = ["p53", "MDM2_nuc", "MDM2_cyt", "PTEN"]

    plt.figure()
    for idx, label in enumerate(labels):
        final_value = data[-1, idx]  # ostatnia wartość danej zmiennej
        label_with_value = f"{label} (end={final_value:.2f})"
        plt.plot(time, data[:, idx], label=label_with_value)

    plt.xlabel("Czas [minuty]")
    plt.ylabel("Ilość cząsteczek")
    plt.title(f"Symulacja: {title}")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"symulacja_{title}.png")
    plt.close()

# główna funkcja
def run_all():
    time_minutes = np.arange(0, 2880, 0.5)
    for scenario, config in BIO_SCENARIOS.items():
        y_initial = config["initial"]
        params = config["params"]
        flags = config["conditions"]
        result = simulate(biological_dynamics, y_initial, time_minutes, args=(params, flags))
        generate_plot(time_minutes, result, scenario)

if __name__ == "__main__":
    run_all()

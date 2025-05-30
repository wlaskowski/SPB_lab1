# Sensitivity Analysis for p53 Protein Level

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

# --- 1. Model biologiczny ---
def model(y, t, params):
    p53, mdm2, mdm2_cyto, pten = y
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']
    d1 = params['d1']
    d2 = params['d2']
    d3 = params['d3']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']

    prod_p53 = p1
    deg_p53 = d1
    prod_mdm2 = p2
    deg_mdm2 = d2
    prod_pten = p3
    deg_pten = d3
    mdm2_transport = k1
    mdm2_thresh = k2
    pten_reg_thresh = k3

    dna_damage = 1
    siRNA = 1 if params.get('siRNA', False) else 0
    pten_active = 1 if params.get('pten_active', True) else 0

    dp53 = prod_p53 + dna_damage - deg_p53 * p53 * (1 + mdm2)
    dmdm2 = prod_mdm2 * (p53 / (mdm2_thresh + p53)) - deg_mdm2 * mdm2 + mdm2_transport * mdm2_cyto
    dmdm2_cyto = prod_mdm2 * (p53 / (mdm2_thresh + p53)) - mdm2_transport * mdm2_cyto - deg_mdm2 * mdm2_cyto
    dpten = prod_pten if pten_active else 0
    dpten -= deg_pten * pten + (siRNA * 0.5)

    return np.array([dp53, dmdm2, dmdm2_cyto, dpten])

# --- 2. RK4 Solver (z laboratorium 1) ---
def rk4(ode_func, y0, t, params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = ode_func(y[i - 1], t[i - 1], params)
        k2 = ode_func(y[i - 1] + h * k1 / 2, t[i - 1] + h / 2, params)
        k3 = ode_func(y[i - 1] + h * k2 / 2, t[i - 1] + h / 2, params)
        k4 = ode_func(y[i - 1] + h * k3, t[i - 1] + h, params)
        y[i] = y[i - 1] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

# --- 3. Scenariusze ---
def apply_scenario(base_params, scenario):
    params = base_params.copy()
    if not scenario['dna_damage']:
        params['d2'] *= 0.1
    if scenario['siRNA']:
        params['p2'] *= 0.02
        params['siRNA'] = True
    else:
        params['siRNA'] = False
    if not scenario['pten_active']:
        params['p3'] = 0
        params['pten_active'] = False
    else:
        params['pten_active'] = True
    return params

# --- 4. Parametry bazowe ---
base_params = {
    "p1": 8.8,
    "p2": 440,
    "p3": 100,
    "d1": 1.375e-14,
    "d2": 0.0001375,
    "d3": 0.00003,
    "k1": 0.0001925,
    "k2": 100000,
    "k3": 150000
}

# --- 5. Definicja scenariuszy ---
scenarios = {
    "A": {"dna_damage": False, "siRNA": False, "pten_active": True},
    "C": {"dna_damage": True, "siRNA": False, "pten_active": False},
}

# --- 6. Symulacja modelu ---
def simulate(params, T=48):
    t = np.linspace(0, T, T+1)
    y0 = [10, 10, 10, 10]
    sol = rk4(model, y0, t, params)
    return t, sol[:, 0]  # zwracamy tylko p53

# --- 7. Lokalna analiza wrażliwości ---
def local_sensitivity_analysis(base_params, scenario_key):
    scenario = scenarios[scenario_key]
    base_applied = apply_scenario(base_params, scenario)
    t, base_result = simulate(base_applied)
    base_final = base_result[-1]

    sens_mean = {}
    sens_final = {}

    for param in base_params:
        dp = 0.01 * base_params[param]
        perturbed = base_params.copy()
        perturbed[param] += dp
        perturbed_result = simulate(apply_scenario(perturbed, scenario))[1]

        dmean = np.mean((perturbed_result - base_result) / dp)
        dfinal = (perturbed_result[-1] - base_final) / dp

        sens_mean[param] = abs(dmean)
        sens_final[param] = abs(dfinal)

    norm_mean = {k: v / sum(sens_mean.values()) for k, v in sens_mean.items()}
    norm_final = {k: v / sum(sens_final.values()) for k, v in sens_final.items()}
    return norm_mean, norm_final

# --- 8. Własna implementacja Sobola (przybliżona wariancja 1. rzędu) ---
def global_sobol_analysis(base_params, scenario_key, N=1000):
    params = list(base_params.keys())
    bounds = {k: (0.8 * v, 1.2 * v) for k, v in base_params.items()}
    scenario = scenarios[scenario_key]

    Y_base = []
    samples = []
    for _ in range(N):
        s = {k: np.random.uniform(*bounds[k]) for k in params}
        samples.append(s)
        _, y = simulate(apply_scenario(s, scenario))
        Y_base.append(y[-1])

    var_Y = np.var(Y_base)
    S = {}

    for param in params:
        Yi = []
        for s in samples:
            s_mod = s.copy()
            s_mod[param] = np.random.uniform(*bounds[param])
            _, y = simulate(apply_scenario(s_mod, scenario))
            Yi.append(y[-1])
        S[param] = abs(np.corrcoef(Y_base, Yi)[0, 1]) ** 2  # przybliżony Sobol 1 rzędu

    norm_sobol = {k: v / sum(S.values()) for k, v in S.items()}
    return norm_sobol

# --- 9. Wykresy ---
def plot_p53_over_time(params, label, filename):
    t, p53 = simulate(params)
    plt.figure()
    plt.plot(t, p53, label=label)
    plt.xlabel('Czas [h]')
    plt.ylabel('Poziom p53')
    plt.title(label)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_single_param_sensitivity(scenario_key, param, direction, method):
    delta = 0.2
    modified = base_params.copy()
    modified[param] *= (1 + direction * delta)
    params = apply_scenario(modified, scenarios[scenario_key])
    label = f"{method}: {param} {int(direction * 100)}% ({scenario_key})"
    filename = f"output/{method}_{scenario_key}_{param}_{int(direction*100)}.png"
    plot_p53_over_time(params, label, filename)

# --- 10. Główna funkcja ---
def main():
    os.makedirs("output", exist_ok=True)

    for scenario_key in scenarios:
        print(f"\nScenariusz: {scenario_key}")

        # Lokalna analiza
        l_mean, l_final = local_sensitivity_analysis(base_params, scenario_key)
        print("\nLokalna analiza – ranking średni (0-48h):")
        for k, v in sorted(l_mean.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v:.4f}")
        print("\nLokalna analiza – ranking końcowy (48h):")
        for k, v in sorted(l_final.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v:.4f}")

        # Globalna analiza
        g_final = global_sobol_analysis(base_params, scenario_key)
        print("\nGlobalna analiza (Sobol) – ranking przybliżony:")
        for k, v in sorted(g_final.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v:.4f}")

        # Wykresy funkcji wrażliwości (top1 i bottom1 lokalna)
        top1 = max(l_mean, key=l_mean.get)
        bottom1 = min(l_mean, key=l_mean.get)
        for param in [top1, bottom1]:
            for direction in [-1, 1]:
                plot_single_param_sensitivity(scenario_key, param, direction, method="local")

        # Wykresy globalne (top1 i bottom1 sobol)
        top1_g = max(g_final, key=g_final.get)
        bottom1_g = min(g_final, key=g_final.get)
        for param in [top1_g, bottom1_g]:
            for direction in [-1, 1]:
                plot_single_param_sensitivity(scenario_key, param, direction, method="global")

if __name__ == "__main__":
    main()

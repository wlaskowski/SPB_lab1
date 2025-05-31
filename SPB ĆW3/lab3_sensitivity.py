import numpy as np
import matplotlib.pyplot as plt
import os

# słownik z głównymi parametrami modelu biologicznego – opisują procesy produkcji i degradacji białek
PARAMETERS = {
    'p1': 8.8,       # szybkość produkcji białka p53
    'p2': 440,       # szybkość produkcji MDM2 pod wpływem p53
    'p3': 100,       # szybkość produkcji PTEN pod wpływem p53
    'd1': 1.375e-14, # współczynnik degradacji p53 (zależny od MDM2 w jądrze)
    'd2': 1.375e-4,  # współczynnik degradacji MDM2
    'd3': 3e-5,      # współczynnik degradacji PTEN
    'k1': 1.925e-4,  # szybkość transportu MDM2 do jądra
    'k2': 1e5,       # stała nasycenia aktywacji transkrypcji MDM2/PTEN
    'k3': 1.5e5      # hamowanie transportu MDM2 przez PTEN
}

# dodatkowe zmienne modyfikujące działanie modelu przy terapii siRNA lub uszkodzeniu DNA
FACTOR = {
    'siRNA_reduction': 0.02,  
    'DNA_repair': 0.1         
}

# zmiany w czasie
# główna funkcja modelująca, jak zmieniają się poziomy p53, MDM2 (cyto i jądro), PTEN w czasie
def biological_process(state, time, config, param):
    p53, mdm_c, mdm_n, pten = state

    dna_intact = config['dna_ok']
    siRNA_active = config['siRNA']
    pten_enabled = config['pten_active']

    # produkcja p53 jest stała
    p53_synthesis = param['p1']
    # degradacja p53 zależy od MDM2 w jądrze
    p53_degradation = param['d1'] * p53 * mdm_n**2
    dp53 = p53_synthesis - p53_degradation

    # modyfikatory zależne od siRNA i stanu DNA
    siRNA_mod = FACTOR['siRNA_reduction'] if siRNA_active else 1.0
    dna_mod = FACTOR['DNA_repair'] if dna_intact else 1.0

    # produkcja MDM2 zależna od p53 i siRNA
    mdm_prod = param['p2'] * siRNA_mod * (p53**4) / ((p53**4) + param['k2']**4)
    # transport MDM2 z cytoplazmy do jądra, hamowany przez PTEN
    mdm_transport = param['k1'] * param['k3']**2 / (param['k3']**2 + pten**2) * mdm_c
    # degradacja MDM2 w cytoplazmie
    mdm_c_deg = param['d2'] * dna_mod * mdm_c
    dmdm_c = mdm_prod - mdm_transport - mdm_c_deg

    # degradacja MDM2 w jądrze
    mdm_n_deg = param['d2'] * dna_mod * mdm_n
    dmdm_n = mdm_transport - mdm_n_deg

    # produkcja PTEN tylko jeśli aktywny
    pten_gen = param['p3'] * (p53**4) / ((p53**4) + param['k2']**4) if pten_enabled else 0.0
    # degradacja PTEN
    dpten = pten_gen - param['d3'] * pten

    return np.array([dp53, dmdm_c, dmdm_n, dpten])

# RK4
# metoda numeryczna (Runge-Kutta 4. rzędu) do rozwiązywania równań różniczkowych
def rk4_solver(equation, y0, time_array, args):
    result = np.zeros((len(time_array), len(y0)))
    result[0] = y0
    for i in range(1, len(time_array)):
        h = time_array[i] - time_array[i - 1]
        t = time_array[i - 1]
        y = result[i - 1]
        k1 = equation(y, t, *args)
        k2 = equation(y + 0.5 * h * k1, t + 0.5 * h, *args)
        k3 = equation(y + 0.5 * h * k2, t + 0.5 * h, *args)
        k4 = equation(y + h * k3, t + h, *args)
        result[i] = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return result

# symulacja
def run_simulation(initial, timeline, scenario, parameter_set):
    return rk4_solver(biological_process, initial, timeline, (scenario, parameter_set))

# liczenie lokalnej wrażliwości
# funkcja licząca lokalną czułość --> czyli jak zmiana parametru wpływa na wynik)
def local_sensitivity_analysis(param_name, init_state, times, base_params, config, delta=0.01):
    mod_params_up = base_params.copy()
    mod_params_down = base_params.copy()
    mod_params_up[param_name] *= (1 + delta)
    mod_params_down[param_name] *= (1 - delta)

    p53_up = run_simulation(init_state, times, config, mod_params_up)[:, 0]
    p53_down = run_simulation(init_state, times, config, mod_params_down)[:, 0]

    return (p53_up - p53_down) / (2 * delta * base_params[param_name])

# globalna analiza wrażliwości
# funkcja losuje wiele zestawów parametrów i sprawdza, który najbardziej wpływa na wynik
def sobol_like_global_analysis(init_state, times, base_params, scenario_flags, trials=500):
    final_vals = []
    param_sets = []
    for _ in range(trials):
        sample = {key: np.random.uniform(0.8 * val, 1.2 * val) for key, val in base_params.items()}
        sim_result = run_simulation(init_state, times, scenario_flags, sample)
        final_vals.append(sim_result[-1, 0])
        param_sets.append(sample)

    final_vals = np.array(final_vals)
    scores = {}
    for key in base_params:
        sampled_vals = np.array([s[key] for s in param_sets])
        if np.std(sampled_vals) > 0 and np.std(final_vals) > 0:
            scores[key] = np.corrcoef(sampled_vals, final_vals)[0, 1] ** 2
        else:
            scores[key] = 0
    return scores

# globalna analiza w czasie
# jak globalna analiza, ale wynik jest sprawdzany dla każdej chwili czasu osobno
def sobol_like_global_time_series(init_state, times, base_params, config, trials=100):
    n_timepoints = len(times)
    results = np.zeros((trials, n_timepoints))
    param_samples = []

    for i in range(trials):
        sample = {key: np.random.uniform(0.8 * val, 1.2 * val) for key, val in base_params.items()}
        sim = run_simulation(init_state, times, config, sample)
        results[i] = sim[:, 0]  # interesuje nas p53
        param_samples.append(sample)

    scores = {key: [] for key in base_params}
    for t_idx in range(n_timepoints):
        p53_vals = results[:, t_idx]
        for key in base_params:
            sampled = np.array([s[key] for s in param_samples])
            if np.std(sampled) > 0 and np.std(p53_vals) > 0:
                score = np.corrcoef(sampled, p53_vals)[0, 1] ** 2
            else:
                score = 0
            scores[key].append(score)
    return scores

# wpływ +/- 20% zmian parametru na p53
def perturbation_response(param, init, times, base_params, flags):
    outputs = {}
    for scale in [0.8, 1.0, 1.2]:
        mod = base_params.copy()
        mod[param] *= scale
        outputs[scale] = run_simulation(init, times, flags, mod)[:, 0]
    return outputs

# wykresy
def plot_sensitivity_curve(times, sensitivities, title, out_path):
    plt.figure()
    for label, series in sensitivities.items():
        plt.plot(times, series, label=label)
    plt.title(f"Zmiana czułości p53 względem parametrów\n{title}")
    plt.xlabel("Czas symulacji [minuty]")
    plt.ylabel("Względna czułość p53")
    plt.legend(title="Parametry")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_param_variation(times, variations, param, scen_name, mode, tag, output_file):
    plt.figure()
    for scale, curve in variations.items():
        label = "wartość bazowa" if np.isclose(scale, 1.0) else f"{scale:.0%} wartości bazowej"
        plt.plot(times, curve, label=label, linestyle='--' if scale != 1.0 else '-')
    plt.title(f"Wpływ zmiany parametru '{param}' na poziom p53\nScenariusz: {scen_name}, analiza: {mode}")
    plt.xlabel("Czas symulacji [minuty]")
    plt.ylabel("Stężenie białka p53")
    plt.legend(title="Wariant parametru")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    t_vals = np.linspace(0, 48*60, 1000)  # symulacja od 0 do 48h, co kilka minut
    y0 = [20000, 120000, 30000, 40000]              # początkowe stężenia p53, MDM2 (cyto), MDM2 (nukleo), PTEN

    # definicja dwóch scenariuszy: zdrowa i rakowa komórka
    SCENARIOS = {
        "normal": {'siRNA': False, 'pten_active': True, 'dna_ok': True},
        "cancerous": {'siRNA': False, 'pten_active': False, 'dna_ok': False}
    }

    for name, flags in SCENARIOS.items():
        print(f"\n--- Scenariusz: {name} ---")
        mean_impact = {}
        terminal_impact = {}

        # analiza lokalna: średnia i końcowa czułość
        for param in PARAMETERS:
            local_resp = local_sensitivity_analysis(param, y0, t_vals, PARAMETERS, flags)
            mean_impact[param] = np.mean(np.abs(local_resp))
            terminal_impact[param] = np.abs(local_resp[-1])

        print("\nLokalna średnia:")
        for k, v in sorted(mean_impact.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.2e}")

        print("\nLokalna końcowa (48h):")
        for k, v in sorted(terminal_impact.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.2e}")

        # analiza globalna końcowa
        global_vals = sobol_like_global_analysis(y0, t_vals, PARAMETERS, flags)
        print("\nGlobalna (Sobol podobna):")
        for k, v in sorted(global_vals.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.4f}")

        # wykresy lokalne (top i bottom parametry)
        top_local = max(mean_impact, key=mean_impact.get)
        bottom_local = min(mean_impact, key=mean_impact.get)
        sensitivities = {
            f"Największy ({top_local})": local_sensitivity_analysis(top_local, y0, t_vals, PARAMETERS, flags),
            f"Najmniejszy ({bottom_local})": local_sensitivity_analysis(bottom_local, y0, t_vals, PARAMETERS, flags)
        }
        plot_sensitivity_curve(t_vals, sensitivities, f"Funkcje wrażliwości - {name}", f"plots/{name}_sensitivity.png")

        for param in [top_local, bottom_local]:
            var = perturbation_response(param, y0, t_vals, PARAMETERS, flags)
            plot_param_variation(t_vals, var, param, name, "lokalna", "zmiana", f"plots/{name}_local_{param}_change.png")

        # wykresy globalne (top i bottom)
        top_global = max(global_vals, key=global_vals.get)
        bottom_global = min(global_vals, key=global_vals.get)

        for param in [top_global, bottom_global]:
            var = perturbation_response(param, y0, t_vals, PARAMETERS, flags)
            plot_param_variation(t_vals, var, param, name, "globalna", "zmiana", f"plots/{name}_global_{param}_change.png")

        # analiza Sobola w czasie
        sobol_series = sobol_like_global_time_series(y0, t_vals, PARAMETERS, flags)
        top_global_series = sobol_series[top_global]
        bottom_global_series = sobol_series[bottom_global]
        plot_sensitivity_curve(
            t_vals,
            {
                f"Największy (globalny {top_global})": top_global_series,
                f"Najmniejszy (globalny {bottom_global})": bottom_global_series
            },
            f"Globalna czułość Sobola - {name}",
            f"plots/{name}_global_sensitivity.png"
        )

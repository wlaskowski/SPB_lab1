import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# parametry
params_nominal = {
    'p1': 8.8,
    'p2': 440,
    'p3': 100,
    'd1': 1.375e-14,
    'd2': 1.375e-4,
    'd3': 3e-5,
    'k1': 1.925e-4,
    'k2': 1e5,
    'k3': 1.5e5
}
param_names = list(params_nominal.keys())

# ustala przedział czasowy symulacji (0-48h) i punkty pomiarowe
t_span = [0, 48]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# ddwa scenariusze
scenarios = {
    "scenariusz_1_zdrowe": {
        "siRNA": 0.02,
        "PTEN_off": 0,
        "no_DNA_damage": 0.1
    },
    "scenariusz_3_uszkodzone": {
        "siRNA": 0.02,
        "PTEN_off": 1,
        "no_DNA_damage": 1.0
    }
}

#  model ODE 
def model(t, y, params, inputs):
    p53, MDM2 = y
    p1, p2, p3 = params['p1'], params['p2'], params['p3']
    d1, d2, d3 = params['d1'], params['d2'], params['d3']
    k1, k2, k3 = params['k1'], params['k2'], params['k3']
    siRNA = inputs['siRNA']
    PTEN_off = inputs['PTEN_off']
    no_DNA_damage = inputs['no_DNA_damage']

    dp53 = p1 - d1 * MDM2 * p53 + k1 * no_DNA_damage
    dMDM2 = p2 * p53 / (p3 + p53) - d2 * MDM2 + k2 * PTEN_off - k3 * siRNA
    return [dp53, dMDM2]

# oblicza lokalną wrażliwość p53 względem parametrów 
def local_sensitivity_with_sens_eqs(params, inputs):
    n_params = len(params)
    y0 = [0, 0]  
    S0 = [0.0] * (2 * n_params)  # początkowe pochodne cząstkowe dy/dp
    yS0 = y0 + S0  # wektor początkowy: stany + wrażliwości

   
    def augmented_system(t, yS):
        y = yS[:2]
        S = np.array(yS[2:]).reshape((2, n_params))
        param_dict = dict(zip(param_names, list(params.values())))

        p53, MDM2 = y
        p1, p2, p3 = param_dict['p1'], param_dict['p2'], param_dict['p3']
        d1, d2, d3 = param_dict['d1'], param_dict['d2'], param_dict['d3']
        k1, k2, k3 = param_dict['k1'], param_dict['k2'], param_dict['k3']
        siRNA = inputs['siRNA']
        PTEN_off = inputs['PTEN_off']
        no_DNA_damage = inputs['no_DNA_damage']

        #  równania modelu
        dp53 = p1 - d1 * MDM2 * p53 + k1 * no_DNA_damage
        dMDM2 = p2 * p53 / (p3 + p53) - d2 * MDM2 + k2 * PTEN_off - k3 * siRNA
        dydt = [dp53, dMDM2]

        # pochodne cząstkowe względem zmiennych (Jacobian)
        dfdx = np.array([
            [-d1 * MDM2, -d1 * p53],
            [p2 * p3 / (p3 + p53)**2, -d2]
        ])

        # pochodne cząstkowe względem parametrów
        dfdp = np.zeros((2, n_params))
        dfdp[0, param_names.index('p1')] = 1
        dfdp[0, param_names.index('d1')] = -MDM2 * p53
        dfdp[0, param_names.index('k1')] = no_DNA_damage
        dfdp[1, param_names.index('p2')] = p53 / (p3 + p53)
        dfdp[1, param_names.index('p3')] = -p2 * p53 / (p3 + p53)**2
        dfdp[1, param_names.index('d2')] = -MDM2
        dfdp[1, param_names.index('k2')] = PTEN_off
        dfdp[1, param_names.index('k3')] = -siRNA

        # oblicza pochodne wrażliwości wg wzoru: dS/dt = dfdx * S + dfdp
        dSdt = np.zeros((2, n_params))
        for i in range(n_params):
            dSdt[:, i] = dfdx @ S[:, i] + dfdp[:, i]

        return dydt + dSdt.flatten().tolist()

    # rozwiązuje rozszerzony układ równań
    sol = solve_ivp(augmented_system, t_span, yS0, t_eval=t_eval)
    # wyodrębnia funkcje wrażliwości p53 względem każdego parametru
    sens_results = {pname: sol.y[2 + i, :] for i, pname in enumerate(param_names)}
    return sens_results, sol.y[0]

# przybliżona globalna analiza
def global_sobol_like(params, inputs, scale=0.2, N=100):
    # tworzy próbki parametrów w zakresie +-20%
    samples = np.random.rand(N, len(params))
    for i, key in enumerate(param_names):
        nominal = params[key]
        samples[:, i] = samples[:, i] * (2 * scale * nominal) + (1 - scale) * nominal

    # oblicza końcową wartość p53 dla każdej próbki
    Y = []
    for row in samples:
        p = dict(zip(param_names, row))
        sol = solve_ivp(model, t_span, [0, 0], args=(p, inputs), t_eval=t_eval)
        Y.append(sol.y[0, -1])
    Y = np.array(Y)

    # liczy wariancję całkowitą i wariancję warunkową dla każdego parametru
    total_var = np.var(Y)
    S1 = []
    for i in range(len(param_names)):
        Xi = samples[:, i]
        bins = np.linspace(np.min(Xi), np.max(Xi), 10)
        means = []
        for j in range(len(bins) - 1):
            mask = (Xi >= bins[j]) & (Xi < bins[j + 1])
            if np.sum(mask) > 5:
                means.append(np.mean(Y[mask]))
        var_i = np.var(means)
        S1.append(var_i / total_var if total_var > 0 else 0)
    return dict(zip(param_names, S1))


os.makedirs("figures_final", exist_ok=True)

# główna pętla
for scen_name, inputs in scenarios.items():
    print(f"\n=== {scen_name.upper()} ===")


    local_sens, base_sol = local_sensitivity_with_sens_eqs(params_nominal, inputs)
    mean_rank = sorted({k: np.mean(np.abs(v)) for k, v in local_sens.items()}.items(), key=lambda x: x[1], reverse=True)
    top_param = mean_rank[0][0]
    last_param = mean_rank[-1][0]
    # nizej wszystkie wykresy
    for pname in [top_param, last_param]:
        plt.figure()
        plt.plot(t_eval, local_sens[pname])
        plt.title(f'{scen_name}: Lokalna wrażliwość p53 względem {pname}')
        plt.xlabel("Czas [h]")
        plt.ylabel("Wrażliwość")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"figures_final/local_sensitivity_{scen_name}_{pname}.png")
        plt.close()



    sobol_result = global_sobol_like(params_nominal, inputs)
    sorted_sobol = sorted(sobol_result.items(), key=lambda x: x[1], reverse=True)
    top_gparam = sorted_sobol[0][0]
    last_gparam = sorted_sobol[-1][0]


    plt.figure()
    plt.bar(sobol_result.keys(), sobol_result.values())
    plt.title(f'{scen_name}: Globalna analiza (wariancja końcowego p53)')
    plt.ylabel('Wskaźnik wrażliwości')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures_final/global_sobol_{scen_name}.png")
    plt.close()


    for pname in [top_param, last_param]:
        plt.figure()
        for factor in [0.8, 1.0, 1.2]:
            p = params_nominal.copy()
            p[pname] *= factor
            sol = solve_ivp(model, t_span, [0, 0], args=(p, inputs), t_eval=t_eval).y[0]
            label = f'{pname} x {factor}'
            plt.plot(t_eval, sol, label=label)
        plt.title(f'{scen_name}: p53 vs {pname} ±20% (LOKALNA)')
        plt.xlabel("Czas [h]")
        plt.ylabel("p53")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"figures_final/local_change_{scen_name}_{pname}.png")
        plt.close()


    for pname in [top_gparam, last_gparam]:
        plt.figure()
        for factor in [0.8, 1.0, 1.2]:
            p = params_nominal.copy()
            p[pname] *= factor
            sol = solve_ivp(model, t_span, [0, 0], args=(p, inputs), t_eval=t_eval).y[0]
            label = f'{pname} x {factor}'
            plt.plot(t_eval, sol, label=label)
        plt.title(f'{scen_name}: p53 vs {pname} ±20% (GLOBALNA)')
        plt.xlabel("Czas [h]")
        plt.ylabel("p53")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"figures_final/global_change_{scen_name}_{pname}.png")
        plt.close()

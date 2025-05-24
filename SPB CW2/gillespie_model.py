import json          
import random        
import math          
import copy          
import matplotlib.pyplot as plt  

# algorytm
def gillespie(model_data, scenario_name, max_time):
    species = model_data["species"]                  # lista białek
    reactions = model_data["reactions"]              # reakcje chemiczne
    base_params = model_data["params"]               # bazowe parametry 
    initial_state = model_data["initial_state"]      # stan początkowy 
    scenarios = model_data["scenarios"]              # scenariusze

    # kopia parametrów, żeby nie nadpisać oryginału
    params = copy.deepcopy(base_params)
    flags = scenarios[scenario_name]

    # modyfikacje zależne od scenariusza
    if not flags["dna_damage"]:  
        params["d2"] *= 0.1      
    if not flags["pten_active"]:  
        params["p3"] = 0.0       
    if flags["siRNA"]:           
        params["p2"] *= 0.02     

    # symulacja
    state = copy.deepcopy(initial_state)             # oecny stan cząsteczek
    time_series = [(0.0, copy.deepcopy(state))]      # lista: czas, stan
    time = 0.0                                        
    steps = 0                                        

    # symulacja trwa dopóki pętla nie przekroczy określonego z góry czasu
    while time < max_time:
        propensities = []  # Lista prawdopodobieństw reakcji --> propensje
        for rxn in reactions:
            local_vars = {**state, **params}  # zmienne stanu, parametry
            try:
                a = eval(rxn["rate"], {}, local_vars)  # obliczanie szybkości reakcji
            except Exception:
                a = 0.0
            propensities.append(max(a, 0.0))  # propensja musi być większa lub równa 0

        a0 = sum(propensities)  # suma propensji
        if a0 <= 0:
            print(f"Propensje spadły do zera na {steps} kroku.")
            time += (max_time - time)  # wtedy --> skok do końca czasu
            time_series.append((time, copy.deepcopy(state)))
            break

        # losowanie czasu do następnej reakcji (tau),  wybóru reakcji
        r1 = random.random()
        r2 = random.random()
        tau = (1.0 / a0) * math.log(1.0 / r1)
        time += tau

        # losujemy która reakcja ma zajść
        threshold = r2 * a0
        cumulative = 0.0
        for i, a in enumerate(propensities):
            cumulative += a
            if cumulative >= threshold:
                selected_rxn = reactions[i]
                break

        # aktualizacja stanu po reakcji
        for sp, count in selected_rxn.get("reactants", {}).items():
            state[sp] -= count
        for sp, count in selected_rxn.get("products", {}).items():
            state[sp] += count

        # zapisywanie stanu i dodanie kroku
        time_series.append((time, copy.deepcopy(state)))
        steps += 1

    # informacja końcowa po symulacji
    print(f"Koniec: czas końcowy = {time:.2f} min, liczba kroków = {steps}")
    print(f"Ostatni stan: {state}")
    return time_series

# wykresy
def plot_realizations(realizations, species, scenario_name):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # kolory

    # dla każdego białka i każdej realizacji rysujemy przebieg czasowy
    for i, sp in enumerate(species):
        for j, result in enumerate(realizations):
            times = [t for t, _ in result]
            values = [state[sp] for _, state in result]
            label = f"{sp} (realizacja {j+1})"
            plt.plot(times, values, label=label, color=colors[i % len(colors)], alpha=0.5 + 0.15 * j)

    plt.title(f"Scenariusz {scenario_name} – wszystkie białka w 3 realizacjach")
    plt.xlabel("Czas [minuty]")
    plt.ylabel("Liczba cząsteczek")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"glsp_scenario_{scenario_name}.png", bbox_inches="tight")
    plt.close()

# uruchomienie
if __name__ == "__main__":
    with open("bio_model.json", "r") as f:
        model_data = json.load(f)  # wczytanie danych z pliku JSON

    # dla każdego scenariusza --> 3 realizacje
    for scenario in model_data["scenarios"].keys():
        print(f"Symulacja scenariusza {scenario} w toku")
        realizations = []
        for i in range(3):
            result = gillespie(model_data, scenario, max_time=2880)  # 48 godzin (2880 minut)
            realizations.append(result)
            print(f"Realizacja {i+1} dla scenariusza {scenario} zakończona.\n")

        # wykres po 3 realizacjach
        plot_realizations(realizations, model_data["species"], scenario)
        print(f"Wykres zapisany do pliku: glsp_scenario_{scenario}.png\n")

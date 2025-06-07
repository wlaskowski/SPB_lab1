import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation  

# parametry symulacji: rozmiar siatki i liczba epok (kroków czasowych)
GRID_SIZE = 65
EPOCHS = 200

# definicje wzorców do użycia w grze
PATTERNS = {
    "glider": np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]), 
    "lwss": np.array([[0,1,1,1,1],[1,0,0,0,1],[0,0,0,0,1],[1,0,0,1,0]]), 
    "diehard": np.array([[0,0,0,0,0,0,1],[1,1,0,0,0,0,0],[0,1,0,0,0,1,1]]), 
    "blinker": np.array([[1, 1, 1]]),  # prosty oscylator o okresie 2
    "toad": np.array([[0, 1, 1, 1], [1, 1, 1, 0]]),  # oscylator o okresie 2
    "acorn": np.array([  # prosty matuzalem
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1]
    ]),
    "glider_gun": np.array([  # działo generujące nieskończoną liczbę gliderów
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ])
}

# funkcja umieszczająca wzorzec w siatce w podanej pozycji
def place_pattern(grid, pattern, top, left):
    rows, cols = pattern.shape  # wymiary wzorca
    grid[top:top+rows, left:left+cols] = pattern  # wstawienie wzorca do siatki

# funkcja zliczająca żywych sąsiadów dla każdej komórki (z użyciem brzegów)
def count_neighbors(grid):
    return sum(np.roll(np.roll(grid, i, 0), j, 1)  # przesunięcia w 8 kierunkach
               for i in (-1, 0, 1) for j in (-1, 0, 1)
               if (i != 0 or j != 0))

# funkcja wykonująca jeden krok gry Conwaya
def step(grid):
    neighbors = count_neighbors(grid)  # liczba sąsiadów
    return (neighbors == 3) | ((grid == 1) & (neighbors == 2))  # reguły --> narodziny, przeżycie, śmierć

# funkcja do zapisu jako GIF
def run_simulation(initial_grid, name="simulacja", epochs=EPOCHS):
    fig, ax = plt.subplots(figsize=(6, 6)) 
    img = ax.imshow(initial_grid, cmap="binary", interpolation="nearest")  
    ax.axis('off') 

    def update(frame):
        nonlocal initial_grid  # pozwala modyfikować siatkę z zewnętrznego zakresu
        initial_grid = step(initial_grid)  # wykonanie kroku gry
        img.set_data(initial_grid)  # aktualizacja obrazu
        return [img]

    anim = FuncAnimation(fig, update, frames=epochs, interval=550, blit=True)  # utworzenie animacji
    anim.save(f"{name}.gif", writer=PillowWriter(fps=2))  # zapisanie animacji jako .gif
    plt.close()  # zamknięcie wykresu

# główna funkcja
def main():
    # symulacja wzorców ciągłego życia (glider, lwss, diehard)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # pusta siatka
    place_pattern(grid, PATTERNS["glider"], 2, 2)  # wstawienie glidera
    place_pattern(grid, PATTERNS["lwss"], 2, 30)  # wstawienie lwss
    place_pattern(grid, PATTERNS["diehard"], 2, 52)  # wstawienie diehard
    run_simulation(grid.copy(), name="ciagle_zycie")  # uruchomienie symulacji

    # symulacja dwóch oscylatorów (blinker i toad)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    place_pattern(grid, PATTERNS["blinker"], 10, 10)
    place_pattern(grid, PATTERNS["toad"], 10, 40)
    run_simulation(grid.copy(), name="oscylatory", epochs=200)

    # symulacja działa (glider gun)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    place_pattern(grid, PATTERNS["glider_gun"], 1, 1)
    run_simulation(grid.copy(), name="dzialo", epochs=200)

    # symulacja matuzalema (acorn)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    place_pattern(grid, PATTERNS["acorn"], 30, 25)
    run_simulation(grid.copy(), name="matuzalem", epochs=600)

# uruchomienie programu
if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

GRID_SIZE = 100

# funkcja zwraca gotowe wzorce do gry (np. blok, oscylator itd.)
def get_patterns():
    return {
        "block": np.array([[1, 1], [1, 1]]),  
        "beehive": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]]),  
        "loaf": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]),  
        "blinker": np.array([[1, 1, 1]]),  # oscylator 
        "toad": np.array([[0, 1, 1, 1], [1, 1, 1, 0]]),  # oscylator
        "r_pentomino": np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])  # matuzalem
    }

# funkcja zwraca działo gliderów 
def get_gosper_glider_gun():
    gun = np.zeros((11, 38), dtype=int)  
    coords = [  # lista współrzędnych, gdzie będą żywe komórki
        (5,1),(5,2),(6,1),(6,2),
        (3,13),(3,14),(4,12),(4,16),(5,11),(5,17),(6,11),(6,15),
        (6,17),(6,18),(7,11),(7,17),(8,12),(8,16),(9,13),(9,14),
        (1,25),(2,23),(2,25),(3,21),(3,22),(4,21),(4,22),
        (5,21),(5,22),(6,23),(6,25),(7,25),
        (3,35),(3,36),(4,35),(4,36)
    ]
    for x, y in coords:
        gun[x, y] = 1  # ustawiamy komórki żywe
    return gun

# klasa GameOfLife 
class GameOfLife:
    def __init__(self, size):
        # tworzymy pustą siatkę 
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

    # metoda do dodania wzorca do planszy w konkretnym miejscu
    def seed_pattern(self, pattern, x, y):
        px, py = pattern.shape  
        self.grid[x:x+px, y:y+py] = pattern 

    # liczymy ilu sąsiadów ma dana komórka 
    def count_neighbors(self, x, y):
        return sum(
            self.grid[(x+i)%self.size, (y+j)%self.size]
            for i in [-1, 0, 1] for j in [-1, 0, 1]
            if not (i == 0 and j == 0)  # pomijamy samą komórkę
        )

    # wykonujemy 1 krok gry == aktualizacja całej planszy
    def step(self):
        new_grid = np.zeros_like(self.grid)  # nowa plansza
        for x in range(self.size):
            for y in range(self.size):
                neighbors = self.count_neighbors(x, y)  # ilu ma sąsiadów

                if self.grid[x, y] == 1 and neighbors in (2, 3):  # żywa komórka przeżywa
                    new_grid[x, y] = 1
                elif self.grid[x, y] == 0 and neighbors == 3:  # martwa ożywa
                    new_grid[x, y] = 1
        self.grid = new_grid  # nadpisujemy starą planszę

    # zapisujemy aktualną planszę 
    def save_frame(self, path, title=""):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap='binary') 
        plt.title(title)
        plt.axis('off')
        plt.savefig(path, format='jpeg')
        plt.close()

    # animacja --> gif
    def animate(self, steps, gif_name, title="Symulacja"):
        fig, ax = plt.subplots(figsize=(6, 6))
        img = ax.imshow(self.grid, cmap='binary') 
        ax.set_title(title)
        ax.axis('off')

        # zapisujemy pierwszą klatkę
        self.save_frame(gif_name.replace(".gif", "_start.jpeg"), title + " - start")


        def update(frame):
            self.step()  # wykonujemy krok
            img.set_data(self.grid)  # aktualizujemy obraz
            return [img]

        # tworzymy animację z ustawioną liczbą kroków
        anim = FuncAnimation(fig, update, frames=steps, interval=300, blit=True)
        anim.save(gif_name, writer=PillowWriter(fps=4))  
        plt.close()

        # zapisujemy ostatnią klatkę
        self.save_frame(gif_name.replace(".gif", "_end.jpeg"), title + " - end")

# funkcja uruchamiająca symulację
def run_simulation(name, patterns_fn, epochs, title):
    life = GameOfLife(GRID_SIZE)  
    for pattern_name, (x, y) in patterns_fn().items(): 
        pattern = get_patterns().get(pattern_name)  
        if pattern is not None:
            life.seed_pattern(pattern, x, y)  
        elif pattern_name == "glider_gun":  
            life.seed_pattern(get_gosper_glider_gun(), x, y)
    life.animate(epochs, name + ".gif", title=title) 


def main():
    run_simulation(
        name="01_zycie_ciagle",
        patterns_fn=lambda: {
            "block": (5, 5),
            "beehive": (10, 30),
            "loaf": (20, 60)
        },
        epochs=15,
        title="Still Lifes"
    )

    run_simulation(
        name="02_oscylatory",
        patterns_fn=lambda: {
            "blinker": (20, 20),
            "toad": (20, 50)
        },
        epochs=20,
        title="Oscylatory"
    )

    run_simulation(
        name="03_dzialo",
        patterns_fn=lambda: {
            "glider_gun": (10, 10)
        },
        epochs=200,
        title="Glider Gun"
    )

    run_simulation(
        name="04_matuzalem",
        patterns_fn=lambda: {
            "r_pentomino": (45, 45)
        },
        epochs=200,
        title="Matuzalem"
    )

if __name__ == "__main__":
    main()

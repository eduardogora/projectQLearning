import pygame
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import os

from mediumMazes import grids, startPoints

pygame.init()

mazeNumber = random.randint(0,4)
mazeNumber = 4
n = len(grids[mazeNumber])
m = len(grids[mazeNumber][0])
print('N: ', n)
print('M: ', m)

grid = grids[mazeNumber]
posI = startPoints[mazeNumber]

HEIGHT, WIDTH = 50*n, 50*m
ROWS, COLS = n, m
CELL_SIZE = WIDTH //  COLS

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Block Fill")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRAY = (0, 200, 200)
GREEN = (0, 255, 0)

#   Constantes
WALL    = 0
PLAYER  = 1
ROAD    = 2
ROAD_C  = 3

# Load configuration
file_name = 'config/gameconfig.json'
with open(file_name, 'r') as file:
    data = json.load(file)

# JSON parameters
fps = data['fps']
max_steps = data['max_steps']
max_episodes = data['max_episodes']
size = data['size']
load_weights = data['load_weights']
enable_render = data['enable_render']

# Clase para manejar el bloque
class Block:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.trail = set()  # Almacena celdas coloreadas

    def move(self, dr, dc):
        new_row = self.row + dr
        new_col = self.col + dc
        # Verifica si la nueva celda es válida y no está coloreada
        if 0 <= new_row < ROWS and 0 <= new_col < COLS and grid[new_row, new_col] != 0 and grid[new_row, new_col] != 3:
            self.row = new_row
            self.col = new_col
            grid[self.row, self.col] = 3  # Marcar como coloreado
            self.trail.add((self.row, self.col))
            print(grid, "\n")

    def draw(self, screen, cont):
        if(cont == 0):
            grid[posI[0]][posI[1]] = 3

        pygame.draw.rect(screen, BLUE, (self.col * CELL_SIZE, self.row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

#   Q-Learning Class

class QlearningGame:
    def __init__(self, size):
        self.q_table = np.zeros((size * size, 4))  # Q-table: (states, actions)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995

    def get_epsilon(self, episode):
        # Decay epsilon over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def take_action(self, state, epsilon):
        # Explore (random action) or exploit (best action)
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

#   Q-Learning Game class

class GAME:
    def __init__(self, load_weights = False, enable_render = True):
        self.run = True
        self.episode = 0
        self.state = 0
        self.steps = 0
        self.wins = 0
        self.reward = 0
        self.epsilon = 0
        self.action = 0
        self.success_history = []
        self.enable_render = enable_render

        # Grid and block
        self.grid = grids[mazeNumber]
        self.block = Block(posI[0], posI[1])  # Initialize the block at the starting position
        self.path = []  # Track the block's path

        self.ql = QlearningGame(size)   #   Q-learning class

    def load_epsilon(self):
        self.epsilon = self.ql.get_epsilon(self.episode)
        if self.episode % 10 == 0 and self.steps == 0:
            print(self.epsilon)
        self.action = self.ql.take_action(self.state, self.epsilon)

    def update_action(self):
        self.prev_row, self.prev_col = self.block.row, self.block.col
        if self.action == 0:    #   left
            new_row, new_col = self.block.row, self.block.col - 1
        elif self.action == 1:  # Right
            new_row, new_col = self.block.row, self.block.col + 1
        elif self.action == 2:  # Up
            new_row, new_col = self.block.row - 1, self.block.col
        elif self.action == 3:  # Down
            new_row, new_col = self.block.row + 1, self.block.col

        # Check if the new position is valid (within bounds and not a wall)
        if 0 <= new_row < ROWS and 0 <= new_col < COLS and self.grid[new_row][new_col] != WALL:
            # Move the block to the new position
            self.block.row, self.block.col = new_row, new_col

            # Mark the new position as visited (colored)
            if self.grid[new_row][new_col] == ROAD:  # Only color if it's a road (not already colored)
                self.grid[new_row][new_col] = ROAD_C  # Mark as colored

            # Add the new position to the path
            if (new_row, new_col) not in self.path:
                self.path.append((new_row, new_col))

    def reward_select(self):
        # Get the current grid position of the block
        grid_row, grid_col = self.block.row, self.block.col

        if self.grid[grid_row][grid_col] == WALL:
            # Penalty for hitting a wall
            self.reward = -10
            # Revert to the previous position
            self.block.row, self.block.col = self.prev_row, self.prev_col
        elif self.grid[grid_row][grid_col] == ROAD:
            # Small reward for coloring a new cell
            self.reward = 1
            # Mark the cell as colored
            self.grid[grid_row][grid_col] = ROAD_C
        elif self.grid[grid_row][grid_col] == ROAD_C:
            # Small penalty for revisiting a colored cell
            self.reward = -1
        else:
            # Default penalty for each step
            self.reward = -1

        # Check if the episode is complete (e.g., all cells are colored)
        if self.is_episode_complete():
            self.reward = 100  # Large reward for completing the episode
            self.episode += 1
            self.wins += 1
            print('WIN')
            print(f'game - Episode {self.episode}, step {self.steps}')
            self.success_history.append(self.steps)
            self.steps = 0
            self.reset_episode()  # Reset the grid and block for the next episode

    def check_step_limit(self):
        if self.steps > max_steps:
            # Reset the episode if the step limit is exceeded
            self.episode += 1
            self.steps = 0
            print(f'game - Episode {self.episode}, step {max_steps}')
            self.reset_episode()  # Reset the grid and block for the next episode

    def is_episode_complete(self):
        # Check if all road cells are colored
        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == ROAD:
                    return False
        return True
    
    def reset_episode(self):
        # Resetear la grilla a un estado en blanco
        self.grid = np.full((ROWS, COLS), ROAD)  # Llenar la grilla con celdas de camino (ROAD)
        
        # Volver a colocar los muros en su lugar
        for row in range(ROWS):
            for col in range(COLS):
                if grids[mazeNumber][row][col] == WALL:
                    self.grid[row][col] = WALL  # Restaurar los muros

        # Reiniciar la posición del bloque
        self.block = Block(posI[0], posI[1])
        
        # Limpiar el historial de la ruta recorrida
        self.path = []

    def increment_step(self):
        self.steps += 1
        if self.enable_render:
            screen.fill(WHITE)  # Limpiar la pantalla
            for row in range(ROWS):
                for col in range(COLS):
                    color = WHITE
                    if self.grid[row][col] == WALL:
                        color = BLACK
                    elif self.grid[row][col] == ROAD_C:
                        color = BLUE
                    pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Dibujar el bloque
            self.block.draw(screen, self.steps)

            pygame.display.flip()

    def run_game(self):
        clock = pygame.time.Clock()

        while self.run and self.episode < max_episodes:
            screen.fill(WHITE)

            # Dibujar la grilla
            for row in range(ROWS):
                for col in range(COLS):
                    color = WHITE if self.grid[row][col] == ROAD else GREEN if self.grid[row][col] == ROAD_C else BLACK
                    pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

            # Dibujar el bloque
            self.block.draw(screen, self.steps)

            # Verificar si se coloreó todo el tablero
            if self.is_episode_complete():
                print("¡Ganaste!")
                self.wins += 1
                self.episode += 1
                self.reset_episode()

            # Ejecutar los pasos del algoritmo Q-learning
            self.load_epsilon()
            self.update_action()
            self.increment_step()
            self.reward_select()
            self.check_step_limit()

            # Actualizar el estado
            next_state = ROWS * self.block.row + self.block.col
            self.ql.update_q_table(self.state, self.action, self.reward, next_state)
            self.state = next_state

            pygame.display.flip()
            clock.tick(30)

        pygame.time.delay(2000)
        pygame.quit()

        # Guardar la Q-table
        if not os.path.exists('Checkpoints'):
            os.makedirs('Checkpoints')
        np.save('Checkpoints/Q_table.npy', self.ql.q_table)
        print(self.ql.q_table)
        print(f'Wins: {self.wins}')

        # Graficar historial de éxito
        plt.plot(self.success_history)
        plt.title('Steps to Reach Goal')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.show()



if __name__ == "__main__":
    game = GAME(load_weights=load_weights, enable_render=enable_render)
    game.run_game()


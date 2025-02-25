import pygame
import numpy as np
import random

#from easyMazes import grids, startPoints
from mediumMazes import grids, startPoints

# Inicializar Pygame
pygame.init()

# Seteo de variables
mazeNumber = random.randint(0, 4)
mazeNumber = 4
n = len(grids[mazeNumber])  # Tamaño del tablero
m = len(grids[mazeNumber][0])  # Tamaño del tablero
print("N: ", n)
print("M: ", m)

# Seteo del mapa
# 0 - Muro
# 1 - Jugador
# 2 - Camino
# 3 - Coloreado

# Generar un tablero básico con caminos (2) y muros (0) en los bordes
grid = grids[mazeNumber]

posI = startPoints[mazeNumber]

# Configuración de la ventana
HEIGHT, WIDTH,  = 50*n, 50*m
ROWS, COLS = n, m
CELL_SIZE = WIDTH // COLS

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Block Fill")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GRAY = (0, 200, 200)
GREEN = (0, 255, 0)

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

# Bucle principal del juego
running = True
clock = pygame.time.Clock()
block = Block(posI[0], posI[1])  # Iniciar en una celda válida


cont = 0

while running:
    screen.fill(WHITE)

    # Dibujar la grilla
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if grid[row, col] == 2 else GREEN if grid[row, col] == 3 else BLACK
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    # Dibujar el bloque
    #block.draw(screen, cont)

    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                block.move(-1, 0)
            elif event.key == pygame.K_DOWN:
                block.move(1, 0)
            elif event.key == pygame.K_LEFT:
                block.move(0, -1)
            elif event.key == pygame.K_RIGHT:
                block.move(0, 1)

    
    # Dibujar el bloque
    block.draw(screen, cont)

    # Verificar si se coloreó todo el tablero
    if np.all((grid == 0) | (grid == 3)):
        print("¡Ganaste!")
        running = False

    pygame.display.flip()
    clock.tick(30)

pygame.time.delay(2000)
pygame.quit()

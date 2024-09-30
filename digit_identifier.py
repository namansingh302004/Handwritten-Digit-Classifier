import pygame
from pygame.locals import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import queue
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'nn_model.h5')

pygame.init()

pygame.display.set_caption("Digit Classification Model")

icon_path = os.path.join(base_dir, 'icon.png') 
if os.path.exists(icon_path):
    icon_image = pygame.image.load(icon_path)
    pygame.display.set_icon(icon_image)

grid_length = 28
cell_size = 20
padding = 1

output_rect_padding = 10
output_rect_color = 'grey'

output_screen_width = 500

width = grid_length * (cell_size + padding) + padding + output_screen_width
height = grid_length * (cell_size + padding) + padding + 20
bg_color = 'black'
screen = pygame.display.set_mode((width, height))

og_cell_color = 'white'
drawn_cell_color = 'black'
smudged_cell_color = 'light grey'


class GridCell:
    none = -1   
    drawn = 1
    smudge = 0.5
    erased = 0

    def __init__(self, x, y) -> None:
        self.surf = pygame.Surface((cell_size, cell_size))
        self.rect = self.surf.get_rect(topleft=(x, y))
        self.surf.fill(og_cell_color)
        self.value = GridCell.erased
        self.state = GridCell.erased
        self.neighbors = []

    def update(self):
        if self.state == GridCell.none:
            return self.value

        if self.state == GridCell.drawn:
            self.surf.fill(drawn_cell_color)
            for cell in self.neighbors:
                if cell.value != GridCell.drawn:
                    cell.state = GridCell.smudge
            self.state = GridCell.none
            self.value = GridCell.drawn
            return self.value

        elif self.state == GridCell.smudge:
            self.surf.fill(smudged_cell_color)
            self.state = GridCell.none
            self.value = GridCell.smudge
            return self.value

        elif self.state == GridCell.erased:
            self.surf.fill(og_cell_color)
            self.state = GridCell.none
            self.value = GridCell.erased
            return self.value

    def draw(self):
        screen.blit(self.surf, self.rect)


class Grid:
    isDrawing = False
    isErasing = False

    def __init__(self) -> None:
       self.grid_cells = [[(x, y) for x in range(grid_length)] for y in range(grid_length)] 
       self.grid_values = [[0 for x in range(grid_length)] for y in range(grid_length)] 
       self.initialise_cells()
    
    def initialise_cells(self):
        for i in range(grid_length):
            for j in range(grid_length):
                cell = GridCell(padding + i * (cell_size + padding), padding + j * (cell_size + padding))
                self.grid_cells[i][j] = cell

        # Set neighbors for each cell
        for i in range(grid_length):
            for j in range(grid_length):
                cell = self.grid_cells[i][j]

                # Assign neighbors
                if i > 0:
                    cell.neighbors.append(self.grid_cells[i-1][j])  # Top neighbor
                if i < grid_length - 1:
                    cell.neighbors.append(self.grid_cells[i+1][j])  # Bottom neighbor
                if j > 0:
                    cell.neighbors.append(self.grid_cells[i][j-1])  # Left neighbor
                if j < grid_length - 1:
                    cell.neighbors.append(self.grid_cells[i][j+1])  # Right neighbor

    def draw(self):
        for i in range(grid_length):
            for j in range(grid_length):
                cell = self.grid_cells[i][j]
                self.grid_values[i][j] = cell.update()
                cell.draw()

    def clear(self):
        for i in range(grid_length):
            for j in range(grid_length):
                cell = self.grid_cells[i][j]
                cell.state = GridCell.erased
                cell.value = GridCell.erased
                self.grid_values[i][j] = 0


class NeuralNetwork:
    def __init__(self, grid) -> None:
        self.model = keras.models.load_model(model_path) 
        self.grid = grid

    def predict(self):
        image_array = np.array(self.grid.grid_values)
        image_array = np.transpose(image_array)
        image_array = image_array.reshape(1, 28, 28)
        prediction = self.model.predict(image_array)
        predicted_digit = np.argmax(prediction[0])
        return prediction[0]


class UI:
    def __init__(self) -> None:
        self.probabilities = [0 for _ in range(10)]
        self.font = pygame.font.SysFont(None, 28)
        self.rect_height = (height - self.font.get_height()) // 10
        self.rect_width = output_screen_width - 100
        self.padding = padding

        self.digit_surfaces = [self.font.render(f"Digit: {i}", True, (255, 255, 255)) for i in range(10)]
        self.prob_surfaces = [self.font.render(f"Prob: 0.00", True, (255, 255, 255)) for _ in range(10)]

        self.rects = [(width - output_screen_width + self.padding,
                      i * self.rect_height + output_rect_padding) for i in range(10)]

        self.min_rect_width = 10
        self.max_rect_width = self.rect_width

        self.clear_surface = self.font.render('Press C to clear', True, (255, 255, 255))
        self.clear_text_position = (0, height - self.font.get_height())  # Position at the bottom

    def update(self, new_probabilities):
        for i, prob in enumerate(new_probabilities):
            self.probabilities[i] = prob

            prob_text = f"Prob: {prob:.2f}"
            self.prob_surfaces[i] = self.font.render(prob_text, True, (255, 255, 255))

    def draw(self):
        for i, prob in enumerate(self.probabilities):
            rect_width = max(int(prob * self.max_rect_width), self.min_rect_width)
            rect_x, rect_y = self.rects[i]

            pygame.draw.rect(screen, output_rect_color, (rect_x, rect_y, rect_width, self.rect_height - output_rect_padding))

            screen.blit(self.digit_surfaces[i], (rect_x + rect_width + 10, rect_y))  # Digit label
            screen.blit(self.prob_surfaces[i], (rect_x + rect_width + 10, rect_y + 20))  # Probability text

        screen.blit(self.clear_surface, self.clear_text_position)


ui = UI()
grid = Grid()
nn = NeuralNetwork(grid)


prediction_queue = queue.Queue()
prediction_thread = None

def run_prediction():
    output = nn.predict()
    prediction_queue.put(output)  

clock = pygame.time.Clock()
running = True

while running:
    any_change = False

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                for i in range(grid_length):
                    for j in range(grid_length):
                        if grid.grid_cells[i][j].rect.collidepoint(event.pos):
                            grid.grid_cells[i][j].state = GridCell.drawn
                            if grid.grid_cells[i][j].value != GridCell.drawn: any_change = True
                Grid.isDrawing = True
                
            if event.button == 3:  # Right click
                for i in range(grid_length):
                    for j in range(grid_length):
                        if grid.grid_cells[i][j].rect.collidepoint(event.pos):
                            grid.grid_cells[i][j].state = GridCell.erased
                            if grid.grid_cells[i][j].value != GridCell.erased: any_change = True
                Grid.isErasing = True

        if event.type == MOUSEMOTION:
            if Grid.isDrawing or Grid.isErasing:
                for i in range(grid_length):
                    for j in range(grid_length):
                        if grid.grid_cells[i][j].rect.collidepoint(event.pos):
                            if Grid.isDrawing:
                                grid.grid_cells[i][j].state = GridCell.drawn
                                if grid.grid_cells[i][j].value != GridCell.drawn: any_change = True
                            else:
                                grid.grid_cells[i][j].state = GridCell.erased
                                if grid.grid_cells[i][j].value != GridCell.erased: any_change = True
                                
        if event.type == MOUSEBUTTONUP:
            Grid.isErasing = False
            Grid.isDrawing = False

        if event.type == KEYDOWN:
            if event.key == K_c:
                any_change = True
                grid.clear()

        if any_change:
            if prediction_thread is None or not prediction_thread.is_alive():
                prediction_thread = threading.Thread(target=run_prediction)
                prediction_thread.start()

    
    if not prediction_queue.empty():
        output = prediction_queue.get()  
        ui.update(output)  

    screen.fill(bg_color)

    ui.draw()
    grid.draw() 

    clock.tick(360)
    pygame.display.update()

pygame.quit()
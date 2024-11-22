import os
import sys

import neat
import pygame
import pymunk

from constants import (
    CENTER,
    CONFIG_PATH,
    FPS,
    GENERATION_TICKS,
    GROUND_HEIGHT,
    HEIGHT,
    MAX_GENERATIONS,
    RENDER_EVERY,
    SAVE_DIR,
    WIDTH,
)
from ragdoll import Ragdoll

current_generation = 0


def simulate(genomes, config):
    global current_generation
    current_generation += 1

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Balancing AI [Generation {current_generation}]")
    clock = pygame.time.Clock()

    # Initialize pymunk
    space = pymunk.Space()
    space.gravity = 0, -981

    # Create the ground
    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_body.position = (CENTER.x, -GROUND_HEIGHT / 2 + 5)
    ground = pymunk.Poly.create_box(ground_body, (WIDTH, GROUND_HEIGHT))
    ground.friction = 0.8
    space.add(ground_body, ground)

    ragdolls = []
    nets = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        ragdolls.append(Ragdoll(space, CENTER.x, CENTER.y - 100))

    # Main loop
    tick = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        if tick >= GENERATION_TICKS:
            running = False

        # Clear screen
        if tick % RENDER_EVERY == 0:
            screen.fill((255, 255, 255))

        # Get ragdoll action
        for ragdoll, net, genome in zip(ragdolls, nets, genomes):
            # Get output of neural net
            output = net.activate(ragdoll.get_data())
            ragdoll.move(output)

            if tick % RENDER_EVERY == 0:
                ragdoll.draw(screen)  # Draw ragdoll

            # Now give them some fitness
            genome[1].fitness += ragdoll.calculate_fitness()

        # Draw the ground
        if tick % RENDER_EVERY == 0:
            pygame.draw.line(screen, (0, 0, 0), (0, HEIGHT), (WIDTH, HEIGHT), 10)  # Ground

        # Update physics
        space.step(1 / FPS)

        # Update display
        pygame.display.flip()
        clock.tick(FPS)
        tick += 1

    pygame.quit()


if __name__ == "__main__":
    # Make directory to save the files
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load Config
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH
    )

    population = neat.Population(config)
    # population = neat.Checkpointer.restore_checkpoint(f"{SAVE_DIR}/neat-checkpoint-29")
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Add a custom checkpoint saver to trigger every 20 generations
    checkpointer = neat.Checkpointer(generation_interval=10, filename_prefix=f"{SAVE_DIR}/neat-checkpoint-")
    population.add_reporter(checkpointer)

    population.run(simulate, MAX_GENERATIONS)

#!/usr/bin/env python3
from itertools import groupby
from sys import argv, exit

from PIL import Image
from PIL import ImageDraw

import numpy as np

# Parse Command Line Arguments
if len(argv) < 3:
    exit("Usage is render.py <input file> <output file> [cutoff]")

input_file = argv[1]
output_file = argv[2]
cutoff = None if len(argv) <= 3 else float(argv[3])

colors = ['black', 'yellow', 'red', 'green']
ncolors = len(colors)

# Helper Functions
def circle_to_box(center_x, center_y, size):
    return (center_x - size, center_y - size,
            center_x + size, center_y + size)

# Process the file
with open(input_file, 'r') as f:
    # Get first line to find the number of particles and the box size.
    xlo, xhi, ylo, yhi = next(f).split()
    xlo, xhi, ylo, yhi = float(xlo), float(xhi), float(ylo), float(yhi)
    boxsize_x = xhi - xlo
    boxsize_y = yhi - ylo
    
    # Compute cutoff_radius
    cutuff_radius = int(1024 * ((cutoff or 0) / boxsize_x))

    # Parse input file
    frames = []
    file_sections = groupby(f, lambda x: x and not x.isspace())
    frame_sections = (x[1] for x in file_sections if x[0])
    for frame_section in frame_sections:
        # Set up a new frame
        img = Image.new('L', (1024, 1024), 'white')
        drawer = ImageDraw.Draw(img)
        frames.append(img)

        num_swarm = 0
        swarm_sizes = []
        
        swarm_idx = 0
        part_idx = 0

        first = True
        # Paint in the frame
        for line in frame_section:

            if first == True:
                swarm_sizes = line.split()
                num_swarm = int(swarm_sizes[0])
                swarm_sizes = swarm_sizes[1:]
                swarm_sizes = [int(s) for s in swarm_sizes]
                first = False
            else:

                if part_idx >= swarm_sizes[swarm_idx]:
                    part_idx = 0
                    swarm_idx += 1
                
                center_x, center_y = line.split()
                center_x, center_y = float(center_x) - xlo, float(center_y) - ylo
                center_y = boxsize_y - center_y
                
                center_x = int(1024 * (float(center_x) / boxsize_x))
                center_y = int(1024 * (float(center_y) / boxsize_y))

                sizes = swarm_sizes.copy()
                sizes[0] = 1
                
                drawer.ellipse(circle_to_box(center_x, center_y,
                                             4*int(np.sqrt(sizes[swarm_idx])) ),
                               colors[ np.mod(swarm_idx, ncolors) ] )
                drawer.ellipse(circle_to_box(center_x, center_y, 1), 'black')

                part_idx += 1

    frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

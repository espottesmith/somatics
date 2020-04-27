import os
from sys import argv, exit

import numpy as np

from pymatgen.core.structure import Molecule

if len(argv) < 3:
    exit("Usage: convert_trajectory_xyzs.py initial_molecule_file, trajectory, to_dir")

initial_molecule_file = argv[1]
trajectory = argv[2]
to_dir = argv[3]

init_mol = Molecule.from_file(initial_molecule_file)
species = [str(e) for e in init_mol.species]
num_atoms = len(species)

if to_dir not in os.listdir():
    os.mkdir(to_dir)

iteration = 0
num_agent = 0
with open(trajectory) as traj_file:
    lines = traj_file.readlines()
    num_agents = int(lines[1].split()[1])

    for line in lines[2:]:
        if line == "\n":
            iteration += 1
            num_agent = 0
        elif "." not in line:
            continue
        else:
            coords = np.array([float(e) for e in line.strip().split(" ")]).reshape(num_atoms, 3)
            mol = Molecule(species, coords)
            mol.to("xyz", os.path.join(to_dir, "{}_{}.xyz".format(iteration,num_agent)))
            num_agent += 1
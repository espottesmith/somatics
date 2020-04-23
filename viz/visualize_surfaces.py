# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sympy
import random


def muller_brown(x, y, sym=False):
    a = [-200, -100, -170, 15]
    b = [-1, -1, -6.5, 0.7]
    c = [0, 0, 11, 0.6]
    d = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    total = 0
    for i in range(4):
        if sym:
            total += a[i] * sympy.exp(b[i] * (x - x0[i]) ** 2 + c[i] * (x - x0[i]) * (y - y0[i]) + d[i] * (y - y0[i]) ** 2)
        else:
            total += a[i] * np.exp(b[i] * (x - x0[i]) ** 2 + c[i] * (x - x0[i]) * (y - y0[i]) + d[i] * (y - y0[i]) ** 2)

    return total


def halgren_lipscomb(x, y, sym=False):
    return ((x - y) ** 2 - (5/3)**2) ** 2 + 4 * (x * y - 4) ** 2 + x - y


def cerjan_miller(x, y, sym=False):
    if sym:
        return (1 - y ** 2) * x ** 2 * sympy.exp(-x**2) + 0.5 * y ** 2
    else:
        return (1 - y ** 2) * x ** 2 * np.exp(-x**2) + 0.5 * y ** 2


def quapp_wolfe_schlegel(x, y, sym=False):
    return x ** 4 + y ** 4 - 2 * x ** 2 - 4 * y ** 2 + x * y + 0.3 * x + 0.1 * y


def culot_dive_nguyen_ghuysen(x, y, sym=False):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def visualize_surface(f, xmin=-1, xmax=1, ymin=-1, ymax=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    xs, ys = np.meshgrid(x, y)
    zs = f(xs, ys)

    ax.plot_surface(xs, ys, zs, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('E')

    plt.show()


def visualize_trajectory(f, trajectory, xmin=-1, xmax=1, ymin=-1, ymax=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    xs, ys = np.meshgrid(x, y)
    zs = f(xs, ys)

    ax.plot_surface(xs, ys, zs, cmap='rainbow', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('E')

    t_xs = np.array([t[0] for t in trajectory])
    t_ys = np.array([t[1] for t in trajectory])
    t_zs = f(t_xs, t_ys)
    ax.plot(t_xs, t_ys, t_zs, c='k')

    plt.show()

def visualize_multiple_trajectories(f, trajectories, xmin=-1, xmax=1, ymin=-1, ymax=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    xs, ys = np.meshgrid(x, y)
    zs = f(xs, ys)

    ax.plot_surface(xs, ys, zs, cmap='rainbow', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('E')

    for trajectory in trajectories:
        t_xs = np.array([t[0] for t in trajectory])
        t_ys = np.array([t[1] for t in trajectory])
        t_zs = f(t_xs, t_ys)
        ax.plot(t_xs, t_ys, t_zs, c=(random.random(), random.random(), random.random()))

    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.show()

def convert_file_trajectory(filename):

    trajectories = list()
    with open(filename, 'r') as file:
        lines = file.readlines()

        params = lines[0].split(" ")
        num_agents = int(params[0])
        xmin = float(params[1])
        xmax = float(params[2])
        ymin = float(params[3])
        ymax = float(params[4])

        for agent in range(num_agents):
            trajectory = list()
            for line in lines[agent + 1::num_agents]:
                contents = line.split(" ")
                trajectory.append((float(contents[0]), float(contents[1])))

            trajectories.append(trajectory)

    return trajectories
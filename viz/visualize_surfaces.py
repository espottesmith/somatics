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


def custom_gaussian_surface(x, y, sym=False):
    As = [-30.205253993631196, -20.022129206599804, -42.43286760440729, -24.488000376985298,
          -16.38178502506924, -17.931659970449473, -30.172687867231787, -25.769765482329085,
          -34.59404185546781, -45.57505912843033, -38.16563302713016, -28.034344300656088,
          -20.891549415761254, -34.312155865769114, -33.39881120712339, -19.19473437486114,
          -16.71738116255264, -46.63256494822467, -39.794827702634194, -42.82053432374118,
          -15.635807642667956, -16.709495942812985, -21.272484277695902, -29.30137489514817,
          -33.116265508423226, -38.003060751763186, -23.054128768670022, -49.73803237924367,
          -34.26923262274106, -33.728651306846615, -47.714241889677, -18.274674558276732]

    xos = [2.6215623694253836, 1.9980209183151225, 0.8046236220265942, 5.345531191556564,
           3.484451329004145, 3.76427721181411, 3.464584936551692, 6.522318729642946,
           3.9430511508814328, 0.8622269732974823, 0.18828370329337502, 5.662984794715073,
           6.638268482227703, 0.05361155321837309, 2.1567971256434197, 6.630245472523514,
           6.482754414476277, 5.194477558182123, 3.5340281376772493, 1.7831630072987896,
           1.6727259991055288, 2.8573999395612613, 4.155770641850253, 0.2824296949648277,
           6.906047307984611, 2.6247148334458674, 4.908560450805521, 4.935138031554453,
           3.0116107367670932, 4.301861220528249, 5.066577354415495, 5.7081045495998035]

    yos = [2.1087399463292322, 1.7185060808091608, 3.685616729673359, 0.9015054892296878,
           3.9034613848553543, 2.4046776136219226, 0.44394661100813193, 3.492904168688485,
           3.772611442157638, 0.5915605965312896, 5.236356834257642, 4.666297415769863,
           0.7283370914029826, 6.332786788672153, 5.756582027080056, 1.1266097874800982,
           5.812512782415781, 2.175038173598125, 3.9121051539632634, 4.08909368771177,
           6.08805554493429, 6.476677552365661, 2.6067435146254776, 5.372661168202783,
           1.6616094641361974, 4.860005422491781, 0.11466475882564396, 4.484541189220203,
           5.818549284349848, 6.238848305863168, 6.930756176264225, 2.664930861278311]

    sigxs = [0.2911656314425136, 0.4691130404784409, 0.7396747576463751, 0.6433880144199259,
             0.5909373096731572, 0.6322752791483168, 0.32412159443916283, 0.4951936022459547,
             0.47789017265910283, 0.6122852231645772, 0.6183325785635645, 0.527027383205543,
             0.48559196388877485, 0.4568484804928763, 0.6011021719280183, 0.5104885333662409,
             0.4067942213014912, 0.681923449202003, 0.5629544963112261, 0.42000967001760536,
             0.6741707385305671, 0.727903748315411, 0.6928721866676995, 0.2846185524146084,
             0.5087515840220862, 0.6903884536586957, 0.7301799534700948, 0.704144239940449,
             0.5970295550984543, 0.662995768592828, 0.4122201521372674, 0.7268597145520569]

    sigys = [0.4331051248265698, 0.2846084349782931, 0.5860790358931132, 0.5293792624478825,
             0.45694555168254414, 0.45213335584286085, 0.38269824172578876, 0.5191566645053056,
             0.2832823138232842, 0.38368209979396223, 0.4809118165455592, 0.43587068456112604,
             0.6959718433848018, 0.42056921316290635, 0.3210091420087929, 0.7349475418728201,
             0.7154978469500543, 0.4185300207474009, 0.5225846530087397, 0.2703262405519922,
             0.3939390828090635, 0.6857847547792397, 0.2709592227738668, 0.41485433110739656,
             0.4494215685461199, 0.7415811492424093, 0.4032122595147482, 0.741140180160017,
             0.6351697337382068, 0.5719958711961134, 0.5318618504689429, 0.5295739338766623]

    total = 0
    for i in range(32):
        if sym:
            total += As[i] * sympy.exp(-1 * ((x - xos[i])**2/(2 * sigxs[i]**2) + (y - yos[i])**2/(2 * sigys[i]**2)))
        else:
            total += As[i] * np.exp(-1 * ((x - xos[i])**2/(2 * sigxs[i]**2) + (y - yos[i])**2/(2 * sigys[i]**2)))

    return total

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
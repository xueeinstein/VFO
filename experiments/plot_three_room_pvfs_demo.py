# ===============================================
# Three Room:
# 5 x (2 + 2 + 2)
#
# +--+--+--+
# |  |  |  |
# |  |  |  |
# |      G'|
# |  |  |  |
# |  |  | G|
# +--+--+--+
#  r1 r2 r3   :rooms
# ===============================================
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# H, W = 5, 6
H, W = 21, 20
THREE_ROOM = np.zeros([H, W], dtype=np.float32)
G_ = [H // 2, int(W * 2 / 3.)]
G = [H - 1, W - 1]
discount = 0.9
GOAL_REWARD = 1.0


def estimate_steps(start_pos, end_pos):
    y_steps = np.abs(start_pos[0] - end_pos[0])
    x_steps = np.abs(start_pos[1] - end_pos[1])
    return y_steps + x_steps


def pvf_room1n2_to_room3():
    pvf = np.copy(THREE_ROOM)

    # compute value for room 1 and 2
    h, w = THREE_ROOM.shape
    for y in range(0, h):
        for x in range(0, int(w * 2 / 3.)):
            steps = estimate_steps([y, x], G_)
            pvf[y, x] = GOAL_REWARD * (discount ** steps)

    # set constant value for room 3
    pvf[:, int(w * 2 / 3.):w] = GOAL_REWARD
    return pvf


def pvf_room3_to_goal():
    pvf = np.copy(THREE_ROOM)

    # compute value for room 3
    h, w = THREE_ROOM.shape
    for y in range(0, h):
        for x in range(int(w * 2 / 3.), w):
            steps = estimate_steps([y, x], G)
            pvf[y, x] = GOAL_REWARD * (discount ** steps)

    return pvf


def plot(pvf, save_as):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(30, 0)
    y = np.arange(THREE_ROOM.shape[0] + 1)
    x = np.arange(THREE_ROOM.shape[1] + 1)
    X, Y = np.meshgrid(x, y)
    h, w = THREE_ROOM.shape
    pvf_ext = np.zeros([h + 1, w + 1], dtype=np.float32)
    pvf_ext[:h, :w] = pvf
    pvf_ext[-1, :w] = pvf[-1, :]
    pvf_ext[:h, -1] = pvf[:, -1]
    pvf_ext[-1, -1] = pvf[-1, -1]
    Z = pvf_ext
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    ax.set_zlim(0, 1.0)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # code to tune view
    # for angle in range(0, 360):
    #     print(angle)
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

    ax.view_init(30, 220)
    plt.savefig(save_as)
    # plt.show()
    print('Save {}'.format(save_as))


def main():
    pvf_1 = pvf_room1n2_to_room3()
    plot(pvf_1, 'pvf_1.png')
    pvf_2 = pvf_room3_to_goal()
    plot(pvf_2, 'pvf_2.png')


if __name__ == '__main__':
    main()

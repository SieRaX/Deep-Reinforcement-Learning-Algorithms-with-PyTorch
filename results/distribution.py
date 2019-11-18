import gym
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random



if __name__ == '__main__':
    x_desired = []
    y_desired = []
    z_desired = []

    x_achieved = []
    y_achieved = []
    z_achieved = []
    for _ in range(10000):
        env = gym.make("FetchReach-v1")
        s = random.randrange(1, 10000)
        env.seed(s)
        a = env.reset()
        x_d = a["desired_goal"][0]
        y_d = a["desired_goal"][1]
        z_d = a["desired_goal"][2]

        x_a = a["achieved_goal"][0]
        y_a = a["achieved_goal"][1]
        z_a = a["achieved_goal"][2]

        x_desired.append(x_d)
        y_desired.append(y_d)
        z_desired.append(z_d)

        x_achieved.append(x_a)
        y_achieved.append(y_a)
        z_achieved.append(z_a)

    x_achieved = np.array(x_achieved)
    y_achieved = np.array(y_achieved)
    z_achieved = np.array(z_achieved)

    x_desired = np.array(x_desired)
    y_desired = np.array(y_desired)
    z_desired = np.array(z_desired)

    print("The result of distribution")
    print("<< Achieved goal >>")
    print("X axis >> mean: {} std: {}".format(np.mean(x_achieved), np.std(x_achieved)))
    print("Y axis >> mean: {} std: {}".format(np.mean(y_achieved), np.std(y_achieved)))
    print("Z axis >> mean: {} std: {}\n".format(np.mean(z_achieved), np.std(z_achieved)))
    print("<< Desired goal >>")
    print("X axis >> mean: {} std: {}".format(np.mean(x_desired), np.std(x_desired)))
    print("Y axis >> mean: {} std: {}".format(np.mean(y_desired), np.std(y_desired)))
    print("Z axis >> mean: {} std: {}".format(np.mean(z_desired), np.std(z_desired)))

    # mpl.rcParams['legend.fontsize'] = 10
    #
    # fig = plt.figure()
    # ax = fig.gca(projection = '3d')
    # ax.plot(x_desired, y_desired, z_desired, 'o', label = 'achieved goal')
    # ax.legend()

    plt.hist(x_desired)
    plt.show()
    plt.hist(y_desired)
    plt.show()
    plt.hist(z_desired)
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('X')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Y')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Z')

    ax1.hist(x_desired)
    ax2.hist(y_desired)
    ax3.hist(z_desired)

    plt.show()


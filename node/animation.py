from numpy import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# Stolen code from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_orientation(x_orientation, y_orientation, z_orientation):
    """
    Plot orientation vectors for x, y and z direction. Start orientation is black, end orientation is magneta with big arrow
    :param x_orientation: vector with dim (1,3) giving the orientation vector for x-axis
    :param y_orientation: vector with dim (1,3) giving the orientation vector for y-axis
    :param z_orientation: vector with dim (1,3) giving the orientation vector for z-axis
    :return: 0
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Just a point where the orientation vec can originate from.
    ax.plot(0, 0, 0, 'o', color='black')
    for ori in range(0, len(x_orientation)):
        if ori == 0:
            x = Arrow3D([0, x_orientation[ori, 0]], [0, x_orientation[ori, 1]],
                        [0, x_orientation[ori, 2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(x)

            y = Arrow3D([0, y_orientation[ori, 0]], [0, y_orientation[ori, 1]],
                        [0, y_orientation[ori, 2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(y)

            z = Arrow3D([0, z_orientation[ori, 0]], [0, z_orientation[ori, 1]],
                        [0, z_orientation[ori, 2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(z)

            ax.set_xlabel('x', color='r')
            ax.set_ylabel('y', color="g")
            ax.set_zlabel('z', color="b")

            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_zlim([-1.1, 1.1])

            plt.draw()

        elif ori==len(x_orientation)-1:
            x = Arrow3D([0, x_orientation[ori, 0]], [0, x_orientation[ori, 1]],
                        [0, x_orientation[ori, 2]], mutation_scale=40, lw=1, arrowstyle="-|>", color="m")
            ax.add_artist(x)

            y = Arrow3D([0, y_orientation[ori, 0]], [0, y_orientation[ori, 1]],
                        [0, y_orientation[ori, 2]], mutation_scale=40, lw=1, arrowstyle="-|>", color="m")
            ax.add_artist(y)

            z = Arrow3D([0, z_orientation[ori, 0]], [0, z_orientation[ori, 1]],
                        [0, z_orientation[ori, 2]], mutation_scale=40, lw=1, arrowstyle="-|>", color="m")
            ax.add_artist(z)

            ax.set_xlabel('x', color='r')
            ax.set_ylabel('y', color="g")
            ax.set_zlabel('z', color="b")

            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_zlim([-1.1, 1.1])

            plt.draw()
        else:
            x = Arrow3D([0, x_orientation[ori, 0]], [0, x_orientation[ori, 1]],
                        [0, x_orientation[ori, 2]],  mutation_scale=10, lw=1, arrowstyle="-|>", color="r")
            ax.add_artist(x)

            y = Arrow3D([0, y_orientation[ori, 0]], [0, y_orientation[ori, 1]],
                        [0, y_orientation[ori, 2]],  mutation_scale=10, lw=1, arrowstyle="-|>", color="g")
            ax.add_artist(y)

            z = Arrow3D([0, z_orientation[ori, 0]], [0, z_orientation[ori, 1]],
                        [0, z_orientation[ori, 2]],  mutation_scale=10, lw=1, arrowstyle="-|>", color="b")
            ax.add_artist(z)

            ax.set_xlabel('x', color='r')
            ax.set_ylabel('y', color="g")
            ax.set_zlabel('z', color="b")

            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_zlim([-1.1, 1.1])

            plt.draw()
    plt.show()


def update_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i, 0:1], data[iteration][i, 1:2], data[iteration][i, 2:])
    return scatters


def animate(position):
    """
        Creates the 3D figure and animates it with the input data.

    :param position: np.array with dim (samples, 3)
    :return: 0
    """
    x = position[:, 0].tolist()
    y = position[:, 1].tolist()
    z = position[:, 2].tolist()

    # Find first non zero element to skip stationaty period at first of animation
    sum_pos = sum([x, y, z], axis=0)
    first_nonzero = 0
    while sum_pos[first_nonzero] == 0:
        first_nonzero += 1

    data = []
    for i in range(first_nonzero - 2, len(x)):
        data.append(np.array([[x[i], y[i], z[i]]]))

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initialize scatters
    scatters = [ax.scatter(data[0][i, 0:1], data[0][i, 1:2], data[0][i, 2:]) for i in range(data[0].shape[0])]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-0.4, 0.4])
    ax.set_xlabel('X')

    ax.set_ylim3d([-0.4, 0.4])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-0.4, 0.4])
    ax.set_zlabel('Z')

    ax.set_title('Leg motion')

    # Defines view of window. elev is rotation from z axis, elevation is rotation in xy plane
    ax.view_init(elev=-120, azim=167)

    # Make the animation
    ani = animation.FuncAnimation(fig, update_scatters, iterations, fargs=(data, scatters),
                                  interval=1, blit=False, repeat=False)

    plt.show()
    return 0


def plot_position(position, x_orientation, y_orientation, z_orientation, fs, fo):
    # Make time axis
    size = len(position)
    ori_points = [int(n*fs/fo) for n in range(int(size*fo/fs))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, len(position)):
        ax.plot(position[i, 0], position[i, 1], position[i, 2], 'o', color='black')

        if i in ori_points:
            x = Arrow3D([position[i, 0], x_orientation[i, 0] + position[i, 0]],
                        [position[i, 1], x_orientation[i, 1] + position[i, 1]],
                        [position[i, 2], x_orientation[i, 2] + position[i, 2]],
                        mutation_scale=10, lw=1, arrowstyle="-|>", color="r")
            ax.add_artist(x)

            y = Arrow3D([position[i, 0], y_orientation[i, 0] + position[i, 0]],
                        [position[i, 1], y_orientation[i, 1] + position[i, 1]],
                        [position[i, 2], y_orientation[i, 2] + position[i, 2]],
                        mutation_scale=10, lw=1, arrowstyle="-|>", color="g")
            ax.add_artist(y)

            z = Arrow3D([position[i, 0], z_orientation[i, 0] + position[i, 0]],
                        [position[i, 1], z_orientation[i, 1] + position[i, 1]],
                        [position[i, 2], z_orientation[i, 2] + position[i, 2]],
                        mutation_scale=10, lw=1, arrowstyle="-|>", color="b")
            ax.add_artist(z)

        ax.set_xlabel('x', color='r')
        ax.set_ylabel('y', color="g")
        ax.set_zlabel('z', color="b")

        ax.set_xlim([position[:, 0].min(), position[:, 0].max()])
        ax.set_ylim([position[:, 1].min(), position[:, 1].max()])
        ax.set_zlim([position[:, 2].min(), position[:, 2].max()])

        plt.draw()
    plt.show()

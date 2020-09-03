import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


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

    #Find first non zero element to skip stationaty period at first of animation
    sum = np.sum([x,y,z],axis=0)
    first_nonzero = 0
    while sum[first_nonzero] == 0:
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


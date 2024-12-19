import numpy as np


def main():
    '''maze = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])'''
    maze = np.array([[0, 0, 0, 0],
                     [0, 1, 0, 1],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]])
    np.save('Data/Maze/manual_maze_grid.npy', maze)
    source = np.array([0, 0])
    destination = np.array([3, 3])
    location = np.vstack((source, destination))
    np.save('Data/Maze/manual_source_destination.npy', location)


if __name__ == '__main__':
    main()

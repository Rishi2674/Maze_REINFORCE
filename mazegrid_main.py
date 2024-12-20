from Simulation.Utils.utils import setup_parser
from Simulation.maze_simulation import Simulation


def main():
    train_mode = True
    model_type = 1
    render = False
    train_episodes = 2000

    sim = Simulation(args, train_mode, train_episodes, model_type, render)
    sim.run_simulation()
    sim.close_simulation()


if __name__ == '__main__':
    args = setup_parser()
    main()

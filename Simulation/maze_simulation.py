import json
import os
import time
import pygame
import numpy as np
from .agent import Agent
from .envs.maze_env import MazeEnv
from .model.dqntorch import DQNModel, DoubleDQNModel, DuelingDQNModel
from .model.policygradienttorch import REINFORCE
from .Utils.utils import DataVisualization


class Simulation:
    def __init__(self, args, train_mode, train_episodes=100, model_type=0, render=False):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)
        print(f'\n' + '%' * 50 + ' Deep Reinforcement Learning Based MAZE Solver ' + '%' * 50)
        self.data_dir = self.params['DATA_DIR'] + self.params['MAZE_DIR']
        filename = self.params['MAZE_FILENAME']

        if args.newmaze:
            self.is_new_map = True
            print(f'Info: Simulation has been started with New Map')
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

        else:
            self.is_new_map = False
            print(f'Info: Simulation has been started with Old Map')
            if not os.path.exists(self.data_dir):
                print(f'Exception: Data directory is not exists. Unable to load saved maze!!')
                exit(0)
            else:
                file_path = self.data_dir + filename
                if not os.path.isfile(file_path):
                    print(f'Exception: Maze file is not exists. Unable to load saved maze!!')
                    exit(0)
                file_path = (self.data_dir + self.params['LOCATION_FILENAME'])
                if not os.path.isfile(file_path):
                    print(f'Exception: Source Destination file is not exists. Unable to load saved maze!!')
                    exit(0)
        self.env = MazeEnv()
        self.n_obs = self.params['NUM_OBSTACLE']

        if self.is_new_map:
            self.env.generate_maze(self.n_obs)
            np.save(self.data_dir + filename, self.env.maze)
            self.env.generate_src_dst()
        else:
            self.env.maze = np.load(self.data_dir + filename)
            self.env.load_src_dst()
        print("Source: ", self.env.source)
        self.agent = Agent(self.env)
        self.train_mode = train_mode
        self.model_type = 'POLICY' if model_type else 'VALUE'
        self.is_trained = None
        self.is_test_completed = False
        self.render = render
        self.train_episodes = train_episodes
        self.test_episodes = self.params['TEST_EPISODES']

        if self.model_type == 'VALUE':
            if args.dqn:
                self.agent.model_name = 'DQN'
            elif args.ddqn:
                self.agent.model_name = 'Double DQN'
            elif args.dueldqn:
                self.agent.model_name = 'Dueling DQN'
            else:
                print(f'Exception: Model type and passed argument are not matched.')
                exit(0)
        elif self.model_type == 'POLICY':
            if args.ppo:
                self.agent.model_name = 'PPO'
            elif args.reinforce:
                self.agent.model_name = 'REINFORCE'
            else:
                print(f'Exception: Model type and passed argument are not matched.')
                exit(0)
        else:
            print(f'Exception: Select model type based on the passed argument: '
                  f'0 for value based and 1 for policy based')
            exit(0)

        self.train_start_time = None
        self.train_end_time = None
        self.sim_start_time = None
        self.sim_end_time = None
        self.is_env_initialized = False
        self.running = True

    def run_simulation(self):
        self.game_initialize()
        self.sim_start_time = time.time()
        while self.running:
            self.event_on_game_window() if self.render else None
            if self.train_mode and not self.is_trained:
                print(f'=' * 65 + ' Training Phase ' + '=' * 66)
                self.train_start_time = time.time()

                if self.model_type == 'VALUE':
                    result = self.agent.train_value_agent(self.train_episodes, self.render)
                else:
                    result = self.agent.train_policy_agent(self.train_episodes, self.render)

                train_data_visual = DataVisualization(self.train_episodes, result,
                                                      self.agent.model_name, self.model_type)
                train_data_visual.save_data()
                train_data_visual.plot_returns()
                train_data_visual.plot_episode_length()
                train_data_visual.plot_training_error()
                train_data_visual.plot_epsilon_decay() if self.model_type == 'VALUE' else None

                self.train_end_time = time.time()
                elapsed_time = self.train_end_time - self.train_start_time
                print(f'Info: Training has been completed...')
                print(f'Info: Total Completion Time: {elapsed_time:.2f} seconds')
                print(f'-' * 147)
                self.is_trained = True

            if (self.is_trained or not self.train_mode) and not self.is_test_completed:
                print(f'\n' + '=' * 66 + ' Testing Phase ' + '=' * 66)
                self.agent.model_filename = (self.agent.model_name + '_' + str(self.env.maze_size) + 'x' +
                                             str(self.env.maze_size) + '_' + str(self.train_episodes) + '_ep_final.pt')

                if self.model_type == 'VALUE':
                    result = self.agent.test_value_agent(self.test_episodes, self.render)
                else:
                    result = self.agent.test_policy_agent(self.test_episodes, self.render)

                success_rate = result[4]
                success = 100 * np.mean(success_rate)
                print(f'Info: Test Success Rate: {success}')
                self.is_test_completed = True
                break
        print(f'%' * 147)

    def game_initialize(self):
        self.agent.position = np.array(self.env.source)
        self.env.find_path()

        if self.render:
            self.env.env_setup()

        if self.train_mode:
            self.is_trained = False
        else:
            self.is_trained = True

        self.is_env_initialized = True
        print(f'Info: Source: {self.env.source} Destination: {self.env.destination}')
        print(f'-' * 147)

        if self.agent.model_name == 'DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = DQNModel(self.agent.state_size, self.agent.action_size, self.env, self.agent.device)
            print(f'Info: DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Double DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = DoubleDQNModel(self.agent.state_size, self.agent.action_size, self.env,
                                              self.agent.device)
            print(f'Info: Double DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Dueling DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = DuelingDQNModel(self.agent.state_size, self.agent.action_size, self.env,
                                               self.agent.device)
            print(f'Info: Dueling DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'REINFORCE':
            print(f'Info: Model Selected: {self.agent.model_name}')
            self.agent.model = REINFORCE(self.agent.state_size, self.agent.action_size, self.env, self.agent.device)
            print(f'Info: REINFORCE with baseline Model is assigned for the Training and Testing of Agent...')

    def event_on_game_window(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def close_simulation(self):
        pygame.quit() if self.render else None

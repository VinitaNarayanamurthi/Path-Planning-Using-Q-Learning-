import  numpy as np

import gym
from gym import error, spaces, utils
from gym.spaces import Box
from gym.spaces import Discrete

import time
import random
import math

# 'using the Mazelab library function  obtained from https://github.com/zuoxingdong/mazelab'

from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as code # Color to represent each objects - Agent, Goal, free path and obstacle
import matplotlib.pyplot as plt
from mazelab import BaseEnv
from mazelab import VonNeumannMotion  # 'North', 'South', 'West', 'East'  => [[-1, 0], [1, 0], [0, -1], [0, 1]])


'Creating the Maze Environment'

from mazelab.generators import random_maze

x = random_maze(width=10 ,height=10, complexity=.5, density=.5)
print(x)

agent_start_pos  = [[1, 1]]
goal_pos = [[9, 9]]
env_id = 'RandomMaze-v0'


'Specifying a 2d maze where 0 represents the path for traversal and 1 represents the obstacles'
# #
# x = np.array([[0, 0, 1, 1, 1],
#               [0, 0, 1, 1, 1],
#               [1, 0, 0, 1, 1],
#               [1, 1, 0, 0, 0],
#               [1, 1, 1, 1, 0]])
# agent_start_pos = [[0, 0]]
# goal_pos = [[4, 4]]
#
# env_id = '5x5Maze-v0'

# x = np.array([[0,0,0,0,0,0,0,0,0,0],
#   [0,1,1,1,1,1,1,1,1,0],
#   [0,1,0,1,0,0,0,0,1,0],
#   [0,1,0,1,0,0,1,1,1,0],
#   [0,1,0,1,1,1,0,0,1,0],
#   [0,1,0,1,0,0,0,0,1,0],
#   [0,1,0,1,0,1,0,0,1,0],
#   [0,1,0,1,0,1,0,0,1,0],
#   [0,1,1,1,1,1,1,1,1,0],
#   [0,0,0,0,0,0,0,0,0,0]])
# agent_start_pos = [[0, 0]]
# goal_pos = [[9, 9]]
# env_id = '10x10Maze-v0'





path_trace = np.stack(np.where(x == 0), axis=1)
obstacle_trace = np.stack(np.where(x == 1), axis=1)


class CreateMaze(BaseMaze):

    """BaseMaze is the  parent class present in mazelab library"""

    @property
    def size(self):
        m_size = x.shape
        return m_size

    def make_objects(self):
        path = Object('path', 0, code.free, False, path_trace)
        obstacle = Object('obstacle', 1, code.obstacle, True,obstacle_trace)
        agent = Object('agent', 2, code.agent, False, [])
        goal = Object('goal', 3, code.goal, False, [])
        return path, obstacle, agent, goal


class Env_setup(BaseEnv):
    def __init__(self):
        super().__init__()
        self.createmaze = CreateMaze()
        self.direction = VonNeumannMotion()
        #' this vonNeumann motion contains the directions 'N','S','W', 'E'''

        'Defining the observation as well as action space'
        self.observation_space = Box(low=0, high=len(self.createmaze.objects), shape=self.createmaze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.direction))  # total 4 actions

    def step(self, action):
        agent_motion = self.direction[action]
        current_position = self.createmaze.objects.agent.positions[0]
        new_position = [current_position[0] + agent_motion[0], current_position[1] + agent_motion[1]]
        if new_position[1] >= self.createmaze.size[1] or new_position[0] >= self.createmaze.size[0]:
            print('warning: agent about to hit the wall , please take someother action')
            prev_action = action
            while (1):
                new_action = env.action_space.sample()
                if (new_action != prev_action):
                    break
            agent_motion = self.direction[new_action]
            new_position = [current_position[0] + agent_motion[0], current_position[1] + agent_motion[1]]

        valid = self._is_valid(new_position)
        if valid:
            self.createmaze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            'when the agent reaches the goal'
            reward = +100
            done = True
            print('Goal position correct ',new_position)

        elif not valid:
            print('the pos is not valid')
            'when agent hits the obstacle/walls'
            reward = -.1

            done = False
        else:

            'when choosing the right path for navigation'
            reward = -.1
            done = False
        print('before',self.createmaze.objects.agent.positions, 'reward', reward,'done', 'a' if done else 'b')
        return self.createmaze.objects.agent.positions, reward, 'a' if done else 'b', {}


    def reset(self):
        self.createmaze.objects.agent.positions = agent_start_pos
        self.createmaze.objects.goal.positions = goal_pos
        return self.createmaze.objects.agent.positions


    def _is_valid(self, position):
        positve_only = position[0] >= 0 and position[1] >= 0
        inbounds = position[0] < self.createmaze.size[0] and position[1] < self.createmaze.size[1]
        passable = not self.createmaze.to_impassable()[position[0]][position[1]]  # 'whether the state contains obstacles or not'
        return positve_only and inbounds and passable


    def _is_goal(self, position):
        out = False

        if position[0] == goal_pos[0][0] and position[1] == goal_pos[0][1]:
            print("Agent position 0 is: %d and goal pos 0 is: %d" % (position[0], goal_pos[0][0]))
            print("Agent position 1 is: %d and goal pos 1 is: %d" % (position[1], goal_pos[0][1]))
            print("Agent reached goal position")
            out = True
        return out


        # for goal in self.createmaze.objects.goal.positions:
        #     if position[0] == goal_pos[0][0] and position[1] == goal_pos[0][1]:
        #         print("it has reached the goallllll")
        #         out = True
        #         break
        # return out


    def get_image(self):
        return self.createmaze.to_rgb()

gym.envs.register(id=env_id, entry_point=Env_setup, max_episode_steps=20000)
env = gym.make(env_id)
env.reset()

# "Environmental Constants"

maze_determine_size= x.shape
low = np.zeros(len(maze_determine_size), dtype=int)
high = np.array(maze_determine_size, dtype=int) - np.ones(len(maze_determine_size), dtype=int)
env.observation_space = spaces.Box(low, high, dtype=np.int64)
print('observation space parameter : ', env.observation_space)
no_of_grids =  maze_determine_size  # one bucket per grid

# 'Number of discrete actions'
no_of_actions = env.action_space.n  # ["N", "S", "E", "W"]
print('no of actions', no_of_actions)

# 'Bounds for each discrete state'
state_limits = list(zip(env.observation_space.low, env.observation_space.high))
print(list(zip(env.observation_space.low, env.observation_space.high)))
print(maze_determine_size[0])

no_of_epochs = 50000
q_updation_iter = np.prod(maze_determine_size, dtype = int) * 100
goal_reach_count_max = 100



# 'Creating a Q-Table for each state-action pair'
# This is a 3 D q table
q_table = np.zeros(no_of_grids + (no_of_actions,), dtype=float)
print('q table\r\n :',q_table)

################# Non Github Repo Code ####################
################# Obtained the idea from https://github.com/MattChanTK/ai-gym/blob/master/maze_2d/maze_2d_q_learning.py ##############
##################### Q Learning ######################


'Defining Environmental Constants'
min_exploration_rate = 0.001
min_learning_rate = 0.2
decay_rate =  np.prod(maze_determine_size, dtype=float) / 10.0
discount_factor = 0.99
convergence_count = 0

def exploration_rate_func(quantity):
    # Exploration rate is  first set high to 0.8 and then reduced linearly for every increase in epoch
    return max(min_exploration_rate , min(0.8, 1.0 - math.log10((quantity + 1) / decay_rate)))



def learning_rate_func(quantity):
    #  Learning rate is  first set high to 0.8 and then reduced linearly for every increase in epoch
    return max(min_learning_rate, min(0.8, 1.0 - math.log10((quantity + 1) / decay_rate)))




def sel_an_action(state,explore_rate):

      # using Epsilon Greedy strategy

    if random.uniform(0,1) < explore_rate:
        action_selected = env.action_space.sample()
        print('random')
    else:
        max_reward_val = np.argmax(q_table[state])
        action_selected = int(max_reward_val)
        print('max reward')

    return action_selected


def state_conversion_bucket(state):
    # Each state is converted to a bucket list

    state = state[0]  #converting a 2D array to 1D array
    bucket_converted = []

    a = len(state)
    for i in range(a):
        bucket_val = (state[i])
        bucket_converted.append(bucket_val)

    return tuple(bucket_converted)


def initialize_epoch():
    observation = env.reset()
    init_state = state_conversion_bucket(observation)
    rewards_obtained = 0
    return init_state, rewards_obtained



def maze_q_learning():

    env.render()




for epoch in range(1000):

        # print("Goal reached for the %d th time" %convergence_count )
        if convergence_count >5 :
            print('Goal reached !!')
            break
        explore_rate = exploration_rate_func(epoch)
        print('explore rate is : ', explore_rate)
        learning_rate = learning_rate_func(epoch)
        print('learning rate is : ', learning_rate)
        init_state, rewards_ont = initialize_epoch()
        print('iteration no', epoch)

        for n in range(200):
            curr_state = init_state
            goal_reached_temp = 'c'
            action_sel = sel_an_action(curr_state,explore_rate)
            print('action selected', action_sel)
            next_state_in_2d, reward, goal_reached_temp,  _ = env.step(action_sel)  # outcome of the selected action
            print('after',  next_state_in_2d, 'reward', reward, 'goal', goal_reached_temp)
            print('goal reached is', goal_reached_temp)
            goal_reached = True if goal_reached_temp=='a' else False

            next_state = state_conversion_bucket(next_state_in_2d)
            print('State reached now', next_state)

            if goal_reached:
                convergence_count += 1

                # if convergence_count == 10:
                #     break
                break
            # if goal_reached:
            #     convergence_count += 1;

            max_reward_q = np.argmax(q_table[next_state])

            q_table[init_state + (action_sel,)] += learning_rate * (
                    reward + discount_factor * (max_reward_q) - q_table[init_state + (action_sel,)])
            print('q table updation type', (q_table))

            # Setting up for the next iteration
            init_state = next_state

            # To visulize the maze
            env.render()




if __name__ == "__main__":
    maze_q_learning()



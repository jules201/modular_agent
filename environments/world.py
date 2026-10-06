# Environment for the Project
# 8x8 Grid world 

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import gymnasium as gym 

class World():
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_actions=5,
                 size=9,
                 bounds=4,
                 modes=3,
                 field_of_view=3,
                 stat_types=[0, 0],
                 variance=1,
                 radius=2,
                 offset=0,
                 circle=True,
                 stationary=True,
                 poisson_shuffle = False,
                 ep_length=50,
                 resource_percentile=50,
                 rotation_speed=0.01,
                 reward_type='sq_dev',
                 module_reward_type='sq_dev',
                 stat_decrease_per_step=0.005,
                 stat_increase_multiplier=1,
                 initial_stats=[0.5, 0.9],
                 set_point=1,
                 pq=[2, 2],
                 reward_clip=None,
                 reward_scaling=1,
                 mod_reward_scaling=1,
                 mod_reward_bias=1):

        super(World, self).__init__()
        self.size = size  # grid size
        self.stat_types = stat_types  # list of stat types, 0 = symmetric, starts at 0.5, auto-increase, 1 = capped, stochastic loss, rest-increase
        self.n_stats = len(self.stat_types)  # number of features (one for each stat)
        self.fov = field_of_view  # field of view of agent
        self.view_size = 2 * self.fov + 1
        self.bounds = bounds  # range of means for resource patches
        self.modes = modes  # number of resource patches per stat
        self.ep_length = ep_length
        self.history = []  # history for visualization
        self.set_point = set_point  # goal of stats
        self.stationary = stationary
        self.border = -0.02  # colour of border and of agent on grid
        self.resource_percentile = resource_percentile  # resources affect stat above this
        self.thresholds = []  # list of thresholds for each resource
        self.multiplier = modes  # multiplies the resource map by fixed amount to keep peaks relatively constant
        self.well_being_range = 0.2
        self.name = 'ResourceWorld'
        self.reward_type = reward_type
        self.module_reward_type = module_reward_type
        self.variance = variance
        self.radius = radius
        self.offset = offset
        self.circle = circle
        self.rotation_speed = rotation_speed
        self.initial_stats = initial_stats
        self.stat_increase_multiplier = stat_increase_multiplier
        self.heat_map = np.zeros((self.size, self.size))
        self.location_history = []
        self.reward_clip = reward_clip
        self.reward_scaling = reward_scaling
        self.mod_reward_scaling = mod_reward_scaling
        self.mod_reward_bias = mod_reward_bias
        self.clamp = 0
        self.squeeze_rewards = True
        self.poisson_shuffle_mean = 0.02
        self.poisson_shuffle = poisson_shuffle

        # self.action_space = spaces.Discrete(n_actions) # action space
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_stats*self.view_size**2+self.n_stats,))

        self.n_actions = n_actions
        self.n_statedims = self.n_stats * self.view_size ** 2 + self.n_stats

        self.action_dim = n_actions
        self.state_dim = self.n_stats * self.view_size ** 2 + self.n_stats

        self.grid = np.zeros((self.n_stats, self.size, self.size))  # empty world grid
        self.loc = [2, 2]
        # self.grid = np.random.choice([0,1],(n_stats,self.size,self.size),p=[0.8,0.2]) # p gives percent of 1s

        self.p, self.q = pq[0], pq[1]  # homeostatic RL exponents for reward function

        self.stat_decrease_per_step = stat_decrease_per_step  # stat decreases over time
        self.stat_decrease_per_injury = 0.05  # stat decrease per danger (not in current version)
        self.stat_increase_per_recover = 0.1  # stat increases if criteria (not in current version)
        self.clumsiness = 0.2
        self.dead = False  # dead if stat hits 0 (just a read out)

        self.make_stats()  # creates stat variable self.stats
        self.reset_grid()
        self.reset()  # resets world, stats, and randomizes location of agent
        self.original_grid = copy.copy(self.grid)

    ## Reset function

    def reset(self):
        if not self.stationary and not self.poisson_shuffle: self.reset_grid() # if changing world on every episode
        # self.make_stats()
        self.time_step = 0  # resets step timer
        self.dead = False
        self.done = False
        self.new = []
        # self.stats = copy.deepcopy(self.initial_stats) # puts stats back to initial values
        # self.loc = [np.random.randint(self.fov,self.size-self.fov),np.random.randint(self.fov,self.size-self.fov)] # initialize state randomly
        # self.loc = [2,2]
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]  # gets initial agent view
        return self.get_state()

    def uncorrelated_shuffle(self):
        for stat in range(self.n_stats):
            if np.random.poisson(self.poisson_shuffle_mean) == 0: continue
            self.grid[stat,:,:] = self.multiplier*get_n_patches(grid_size = self.size, bounds = self.bounds, modes = self.modes, var = self.variance) # makes gaussian patches
            self.thresholds[stat] = np.percentile(self.grid[stat,:,:],self.resource_percentile)

        self.grid[:,[0,-1],:] = self.grid[:,:,[0,-1]] = self.border # make border

    def correlated_shuffle(self):
        self.thresholds = []
        self.offset += np.random.uniform(0,2*np.pi) #self.rotation_speed #np.random.uniform(0,2*np.pi)
        self.grid = get_n_resources(grid_size = self.size, bounds = self.bounds, resources = self.n_stats, radius = self.radius, offset = self.offset, var = self.variance)
        for stat in range(self.n_stats):
            self.thresholds.append(np.percentile(self.grid[stat,:,:],self.resource_percentile))

        self.grid[:,[0,-1],:] = self.grid[:,:,[0,-1]] = self.border # make border

    def reset_grid(self):
        if not self.circle:
            self.thresholds = []
            for stat in range(self.n_stats):
                self.grid[stat, :, :] = self.multiplier * get_n_patches(grid_size=self.size, bounds=self.bounds,
                                                                        modes=self.modes,
                                                                        var=self.variance)  # makes gaussian patches
                self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        if self.circle:
            self.thresholds = []
            if not self.stationary:
                self.offset += np.random.uniform(0, 2 * np.pi)  # self.rotation_speed #np.random.uniform(0,2*np.pi)
                # self.radius = np.random.uniform(1, 3)
            self.grid = get_n_resources(grid_size=self.size, bounds=self.bounds, resources=self.n_stats,
                                        radius=self.radius, offset=self.offset, var=self.variance)
            for stat in range(self.n_stats):
                self.thresholds.append(np.percentile(self.grid[stat, :, :], self.resource_percentile))

        self.grid[:, [0, -1], :] = self.grid[:, :, [0, -1]] = self.border  # make border

    def change_first_resource_map(self):
        self.grid[0] = get_n_resources(grid_size = self.size, bounds = self.bounds, resources = self.n_stats, radius = 0, offset = 2, var = self.variance)[0]
        self.thresholds[0] = np.percentile(self.grid[0],self.resource_percentile)
        self.grid[:,[0,-1],:] = self.grid[:,:,[0,-1]] = self.border # make border

    def revert_to_original_grid(self):
        self.grid = self.original_grid

    def reset_for_new_agent(self):
        self.make_stats()
        self.loc = [5, 5]
        self.time_step = 0  # resets step timer
        self.dead = False
        self.done = False
        self.history = []
        self.location_history = []
        self.heat_map = np.zeros((self.size, self.size))
        self.revert_to_original_grid()
        self.reset_grid()

    def move_location(self, loc):
        self.loc = loc
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]

    ## Step function

    def step(self, action):
        # Execute one time step within the environment
        self.time_step += 1
        self.set_lowest_stat()

        if self.time_step == self.ep_length: self.done = True

        if action == 0 and self.loc[0] < self.size - self.fov - 1:
            self.loc[0] += 1

        elif action == 1 and self.loc[0] > self.fov:
            self.loc[0] -= 1

        elif action == 2 and self.loc[1] < self.size - self.fov - 1:
            self.loc[1] += 1

        elif action == 3 and self.loc[1] > self.fov:
            self.loc[1] -= 1

        reward = self.step_stats()
        if self.reward_clip is not None: reward = np.clip(reward, -self.reward_clip,
                                                          self.reward_clip)  # reward clipping

        if self.module_reward_type == 'HRRL': module_rewards = self.separate_HRRL_rewards()
        if self.module_reward_type == 'sq_dev': module_rewards = self.separate_squared_rewards()
        if self.module_reward_type == 'lin_dev': module_rewards = self.separate_linear_rewards()

        # self.separate_squared_rewards() #self.separate_HRRL_rewards(100)
        self.view = self.grid[:, self.loc[0] - self.fov: self.loc[0] + self.fov + 1,
                    self.loc[1] - self.fov: self.loc[1] + self.fov + 1]
        self.history.append((self.grid_with_agent(), self.view, copy.deepcopy(self.stats)))
        self.location_history.append(copy.copy(self.loc))
        self.heat_map[self.loc[0], self.loc[1]] += 1

        if self.squeeze_rewards:
            reward = np.tanh(reward)
            module_rewards = np.tanh(module_rewards)

        if not self.stationary and self.poisson_shuffle and not self.circle: self.uncorrelated_shuffle() # shuffles resource maps at poisson rate
        if not self.stationary and self.poisson_shuffle and self.circle: self.correlated_shuffle() # shuffles resource maps at poisson rate

        return self.get_state(), reward, self.done, module_rewards  # self.dead

    ##### Functions involved in making stats, stepping stats and getting HRRL Rewards

    def make_stats(self):
        self.stats = []
        for stat in self.stat_types:
            if stat == 0:
                self.stats.append(np.random.uniform(self.initial_stats[0], self.initial_stats[1]))
            else:
                self.stats.append(np.random.uniform(0.7, 0.9))

        self.lowest_stat = np.min(self.stats)

        # self.initial_stats = copy.deepcopy(self.stats)

    def set_lowest_stat(self):
        if np.min(self.stats) < self.lowest_stat:
            self.lowest_stat = np.min(self.stats)

    def step_stats(self):
        self.old_stats = copy.deepcopy(self.stats)

        for i in range(self.n_stats - self.clamp):

            if self.stat_types[i] == 0:  # for food/water stats
                # if  self.grid[i,self.loc[0],self.loc[1]] == 1: self.stats[i] += self.stat_increase_per_recover
                self.stats[i] -= self.stat_decrease_per_step[i]
                if self.grid[i, self.loc[0], self.loc[1]] > self.thresholds[i]:
                    self.stats[i] += self.stat_increase_multiplier * self.grid[i, self.loc[0], self.loc[1]]

            if self.stat_types[i] == 1:  # damage stats
                # if self.grid[i,self.loc[0],self.loc[1]] == 1 and np.random.uniform() < self.clumsiness: self.stats[i] -= self.stat_decrease_per_injury # stochastic damage
                # if self.action == 4: self.stats[i] += self.stat_increase_per_recover # for rest action
                self.stats[i] += self.stat_decrease_per_step[i]
                self.stats[i] -= self.stat_increase_multiplier * self.grid[i, self.loc[0], self.loc[1]]
                if self.stats[i] > 1: self.stats[i] = 1

            if self.stat_types[i] == 2:
                self.stats[i] = np.random.normal(self.set_point, 1)

            if self.stats[i] < 0:
                # self.stats[i] = 0
                self.dead = True

        if self.reward_type == 'HRRL': return self.HRRL_reward()
        if self.reward_type == 'HRRL_exact_sum': return sum(self.separate_HRRL_rewards())
        if self.reward_type == 'sq_dev': return self.sq_dev_reward()
        if self.reward_type == 'well_being': return self.well_being_reward()
        if self.reward_type == 'lin_dev': return self.lin_dev_reward()
        if self.reward_type == 'min_sq': return self.min_sq()

    def HRRL_reward(self):
        return self.reward_scaling * (self.get_cost_surface(self.old_stats) - self.get_cost_surface(
            self.stats))  # - 100*any([x<0.05 for x in self.stats])

    def sq_dev_reward(self):
        return 0.2 - sum([(self.set_point - stat) ** 2 for stat in self.stats])

    def lin_dev_reward(self):
        return sum([0.2 - np.abs(self.set_point - stat) for stat in self.stats])

    def separate_linear_rewards(self):
        return [0.2 - 4 * np.abs(self.set_point - stat) for stat in self.stats]

    def separate_squared_rewards(self):
        return [0.1 - np.abs(self.set_point - stat) ** 2 for stat in self.stats]

    def separate_HRRL_rewards(self):

        return [self.mod_reward_scaling * ((np.abs(self.set_point - old_stat) ** self.p) ** (1 / self.q) - (
                    np.abs(self.set_point - new_stat) ** self.p) ** (1 / self.q)) - self.mod_reward_bias for
                old_stat, new_stat in zip(self.old_stats, self.stats)]

    def well_being_reward(self):
        return 1 if all(
            [stat > self.set_point - self.well_being_range and stat < self.set_point + self.well_being_range for stat in
             self.stats]) else -1

    def min_sq(self):
        return min([-np.abs(self.set_point - stat) ** 2 for stat in self.stats])

    def get_cost_surface(self, stats):
        return sum([np.abs(self.set_point - stat) ** self.p for stat in stats]) ** (1 / self.q)

    def get_state(self):
        return torch.cat((torch.tensor(self.stats).float(), torch.tensor(self.view.flatten()).float())).float()

    def grid_with_agent(self):
        temp = copy.copy(self.grid)
        temp[:, self.loc[0], self.loc[1]] = self.border  # self.grid[:,self.state[0],self.state[1]]
        # temp[:2][temp[:2]>self.resource_threshold] = 2
        return temp

    def render(self):
        tits = [f'stat {i + 1}' for i in range(self.n_stats)]
        for i in range(self.n_stats):
            plt.subplot(100 + 10 * self.n_stats + 1 + i)
            plt.title(tits[i])
            plt.imshow(self.grid_with_agent()[i])
            plt.xticks([])
            plt.yticks([])
        plt.figure()
        # plt.imshow(self.heat_map)
        # plt.show()




import coordinates
import player as player
import render
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class logic(py_environment.PyEnvironment):
    def __init__(self,player,coordinates,render,generate_video=False,base_reward=50,discount=1.0):
        # Passing the class player object
        self.discount = discount
        self.base_reward = base_reward
        self.player = player
        self.coordinates = coordinates
        self.render = render
        self.num_actions = 4 # I don't think we will change it

        self.num_fram_sinc_start = 0
        self.max_frames = (self.coordinates.row * self.coordinates.col) * 2 # multiply by two to allow it to move the map twice

        self._state = self.coordinates.get_map()
        self.observation_shape = self.coordinates.get_map().shape
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_actions, name='action')
                                        # the minimum will be the map with all are zeros because it was when the coordination object was constructed
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.observation_shape, dtype=np.int32, minimum=self.coordinates.get_min_map(), name='observation')

        self._episode_ended = False

        self.iteration_num = 0
        self.generate_video = generate_video

    # ----------------------------------------------------------------------------

    def _reset(self):
        self.num_fram_sinc_start = 0
        self.coordinates.reset()
        self._state = self.coordinates.get_map()
        self._episode_ended = False

        self.iteration_num += 1
        if self.generate_video:
            self.render.add_frame_to_list(self._state)

        return ts.restart(np.array(self._state, dtype=np.int32))
    # ----------------------------------------------------------------------------

    def _step(self, action):
        if self._episode_ended:
            if self.generate_video:
                self.render.generate_video(self.iteration_num)
                self.render.reset_frame_list()

            return self.reset()

        # Make sure episodes don't go on forever.
        if self.num_fram_sinc_start >= self.max_frames or self._episode_ended:
            self._state, reward, _ = self.game_logic(action)
            reward = -reward * 10
            self._episode_ended = True

            if self.generate_video:
                #print(f"started generating video of {len(self.render.frames_list)}")
                self.render.add_frame_to_list(self._state)
                self.render.generate_video(self.iteration_num)
                self.render.reset_frame_list()

            return ts.termination(np.array(self._state, dtype=np.int32), reward)

        # Game Logic Function Here
        else:
            self._state,reward,discount = self.game_logic(action)
           # print(f"Current reward: {reward}")
            if self.generate_video:
                self.render.add_frame_to_list(self._state)
            return ts.transition(np.array(self._state, dtype=np.int32), reward=reward, discount=discount)
    # ----------------------------------------------------------------------------
    def game_logic(self,action):
        self.num_fram_sinc_start += 1
        map, reward,discount = self.movement_logic(action)

        return map,reward,discount
    # ----------------------------------------------
    def movement_logic(self,action):
        # AI have to decide his first position to start the map
        #print(f"Action taken: {action}")
        if self.num_fram_sinc_start ==1:
            player_pos = self.player.initate_pos(action)
            self.coordinates.add_player_to_map(player_pos)
            reward = 0
            discount = self.discount
            return self.coordinates.get_map(),reward,discount
        elif self.coordinates.num_obj <=0:
            reward = self.base_reward * 100
            discount = self.discount
            return self.coordinates.get_map(),reward,discount
        else:
            player_next_pos = self.player.action_taker(action)
            if self.coordinates.is_obj(player_next_pos):
                self.coordinates.remove_objects_from_map(player_next_pos)
                reward = self.base_reward
                print(f"Got a reward of value: {reward}-----------------------------------------")
                discount = self.discount
                return self.coordinates.get_map(),reward,discount
            else:
                reward = -0.75
                discount = self.discount
                return self.coordinates.get_map(), reward, discount

    # ----------------------------------------------
    #TODO remove it later if not needed
    def reward_logic(self):
        return - self.num_fram_sinc_start * self.coordinates.num_obj
    # ---------------------------
    def action_spec(self):
        return self._action_spec
    # ----------------------------------------------------------------------------
    def observation_spec(self):
        return self._observation_spec
    # -------------------------------------------------
    def enable_generating_video(self):
        self.generate_video = True
    # --------------------------------------
    def disable_generating_video(self):
        self.generate_video = False

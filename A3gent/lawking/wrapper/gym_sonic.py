"""
    desc: atari envionment warpper.
    create: 2017.12.28
    @author: sam.dm
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import numpy as np
from lawking.wrapper.atari_wrapper import wrap_deepmind
from a3gent.environments import Environment

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-lawking environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()

def make_sonic_env(game, state):
    """
    Create an environment with some standard wrappers.
    """
    from retro_contest.local import make

    env = make(game=game, state=state)

    env = SonicDiscretizer(env)

    return env

class GymSonic(Environment):

    def __init__(self, gym_id, frame_stack, clip_rewards, episode_life, wrap_frame, visualize=False):
        """
        Initialize OpenAI Gym Atari.

        Args:
            gym_id: OpenAI Gym atari environment ID. See https://gym.openai.com/envs
            frame_stack: Whether stack lastest 4 frames.
            clip_rewards: Whether clip rewards into [-1,0,1].
            episode_life: Whether finished a episode when lost life.
            wrap_frame: Whether wrap frame as [84, 84, 1] and color from RGB to gray.
            visualize: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probabily going to slow down the training.
        """
        self.gym_id = gym_id
        game_state = gym_id.split('#')
        env = make_sonic_env(game_state[1], game_state[2])
        self.env = wrap_deepmind(env, frame_stack=frame_stack, clip_rewards=clip_rewards,
            episode_life=episode_life, wrap_frame=wrap_frame)
        self.visualize = visualize

    def __str__(self):
        return 'GymAtari({})'.format(self.gym_id)

    def close(self):
        self.env.close()
        self.env = None

    def reset(self):
        return self.env.reset()

    def execute(self, actions):
        if self.visualize:
            self.env.render()
        if isinstance(actions, dict):
            actions = [actions['action{}'.format(n)] for n in range(len(actions))]
        state, reward, terminal, _ = self.env.step(actions)
        return state, terminal, reward

    @property
    def states(self):
        return dict(shape=self.env.observation_space.shape, type='uint8')

    @property
    def actions(self):
        return dict(type='int', num_actions=self.env.action_space.n)
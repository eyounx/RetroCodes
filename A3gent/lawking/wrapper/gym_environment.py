# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from a3gent.environments.meta_environment import MetaEnvironment
import time

from a3gent.contrib.openai_gym import OpenAIGym
from a3gent.environments.vec_env.make_vec_env import MakeVecEnv
from lawking.wrapper.gym_sonic import GymSonic

from a3gent.exception import A3gentError
import sys

no_atari = True
if 'cv2' in sys.modules:
    no_atari = False
    from a3gent.contrib.gym_atari import GymAtari

class GymSimulator(MetaEnvironment):
    """
    GymSimulator Environment, Mostly used in Algorithm validation
    """
    def __init__(self, config):
        config['env_type'] = 'gym'
        super(GymSimulator, self).__init__(config)

        # init simulator
        # default simulate env is CartPole
        self.gym_id = 'CartPole-v0'

        # If set True, the program will visualize the trainings of gym's environment.
        self.visualize = False

        # If set True, output frames will stack by 4 frames.
        self.frame_stack = True

        # If set True, atari reward will be clipped into [-1, 0, 1].
        self.clip_rewards = True

        # If set True, atari will be finshed one episode when lost life.
        self.episode_life = True

        # If set True, all frame will be wrapped as nature paper did.
        self.wrap_frame = True

        # Output Directory. Set this to None disables monitoring.
        self.gym_monitor = None

        # Setting this to True prevents existing log files to be overwritten
        self.gym_monitor_safe = False

        # Save a video every monitor_video steps. 0 disables recording of videos
        self.gym_monitor_video = 0

        # If use vec environment
        self.vec_env = False

        # Number of environments
        self.num_env = 16

        # Environment seed
        self.env_seed = 0

        # Num steps for each envionment rollout
        self.batch_size = 1

        # Batch size
        self.nsteps = 5

        # parse more config
        self.parse_env_config()

        # set flag_stop
        self.flag_stop = False

    def __str__(self):
        return 'OpenAIGym({})'.format(self.gym_id)

    def parse_env_config(self):
        """
        Obtain configuration of gym_id, monitor path and saving video
        """

        if 'gym_id' in self.env_conf:
            self.gym_id = self.env_conf['gym_id']

        if 'frame_stack' in self.env_conf:
            self.frame_stack = self.env_conf['frame_stack']

        if 'clip_rewards' in self.env_conf:
            self.clip_rewards = self.env_conf['clip_rewards']

        if 'episode_life' in self.env_conf:
            self.episode_life = self.env_conf['episode_life']

        if 'wrap_frame' in self.env_conf:
            self.wrap_frame = self.env_conf['wrap_frame']

        if 'vec_env' in self.env_conf:
            self.vec_env = self.env_conf['vec_env']

        if 'num_env' in self.env_conf:
            self.num_env = self.env_conf['num_env']

        if 'env_seed' in self.env_conf:
            self.env_seed = self.env_conf['env_seed']

        if 'nsteps' in self.env_conf:
            self.nsteps = self.env_conf['nsteps']

        _Atari59 = [
            'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix',
            'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider',
            'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
            'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
            'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite',
            'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond', 'JourneyEscape',
            'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
            'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
            'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
            'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
            'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
            'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'
        ]

        # init OpenAIGym
        game_name = self.gym_id.split("-")[0]

        if self.gym_id.startswith('Sonic'):
            env_fn = lambda: GymSonic(
                gym_id=self.gym_id,
                frame_stack=self.frame_stack,
                clip_rewards=self.clip_rewards,
                episode_life=self.episode_life,
                wrap_frame=self.wrap_frame,
                visualize=self.visualize
            )
        elif game_name in _Atari59:
            if no_atari:
                raise A3gentError('Can not find cv2 to run gym_atari{name}'\
                    ', please install cv2 or try another game_id'.format(name=game_name))
            env_fn = lambda: GymAtari(
                gym_id=game_name + "NoFrameskip-v4",
                frame_stack=self.frame_stack,
                clip_rewards=self.clip_rewards,
                episode_life=self.episode_life,
                wrap_frame=self.wrap_frame,
                visualize=self.visualize
            )
        else:
            env_fn = lambda: OpenAIGym(
                gym_id=self.gym_id,
                monitor=self.gym_monitor,
                monitor_safe=self.gym_monitor_safe,
                monitor_video=self.gym_monitor_video,
                visualize=self.visualize
            )

        if self.vec_env:
            self.env = MakeVecEnv(env_fn=env_fn, num_env=self.num_env)()
        else:
            self.env = env_fn()

        # Get state_spec and action_spec
        self.states_spec = self.env.states
        self.actions_spec = self.env.actions

    def execute(self, actions):
        """
        Interact with the environment
        if set interactive to True, env.execute will apply an action to the environment and
        get an observation after the action
        """
        state, terminal, step_reward = self.env.execute(actions=actions)

        return (state, terminal, step_reward)

    def read(self, local_q=None):
        """
        if set interactive to False, call env.read instead of call env.execute
        func read will fetch a batch of data from local_q through a loop . each item will be a
        tuple of (state, action, terminal, reward, next_state,[internal, next_internal])
        """
        if local_q is None:
            raise A3gentError('Need A local Queue to do environment execute in mode of non-interaction')
        states = list()
        internals = list()
        actions = list()
        terminals = list()
        rewards = list()
        next_states = list()
        next_internals = list()
        for _ in range(self.batch_size):
            elements_tuple = local_q.pop()
            # TODO: add a func of "is_empty()" in local_q
            wait_time = 0
            while elements_tuple is None:
                time.sleep(1)
                elements_tuple = local_q.pop()
                wait_time += 1
                if wait_time > 30:
                    print('wait time reach max {max_t}'.format(max_t=wait_time))
                    self.flag_stop = True
                    return None

            if len(elements_tuple) == 5:
                state, action, terminal, reward, next_state = elements_tuple
                internal = list()
                next_internal = list()
            elif len(elements_tuple) == 7:
                state, action, terminal, reward, next_state, internal, next_internal = elements_tuple
            else:
                raise A3gentError('Invalid elment_tuple from localQueue.pop, elment_tuple must consist of ' \
                                  '(state, action, terminal, reward, next_state) or (state, action, terminal, reward, next_state, internal, next_internal)')
            states.append(state)
            actions.append(action)
            terminals.append(terminal)
            rewards.append(reward)
            next_states.append(next_state)
            for n, internal_ in enumerate(internal):
                if len(internals) <= n:
                    internals.append(list())
                internals[n].append(internal)
            for n, next_internal_ in enumerate(next_internal):
                if len(next_internals) <= n:
                    next_internals.append(list())
                next_internals[n].append(next_internal_)

        return (dict(state=states), internals, dict(action=actions), terminals, rewards, dict(state=next_states), next_internals)


    def should_stop(self):
        return self.flag_stop

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        """
        Call reset only in interactive mode
        """
        return self.env.reset()

    def close(self):
        self.env.close()

    @property
    def states(self):
        return self.states_spec

    @property
    def actions(self):
        return self.actions_spec

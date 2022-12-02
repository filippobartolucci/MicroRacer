
# EnvWrapper

# ---------------------------
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import tracks


class MicroRacerEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=-1, maximum=1 , name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum= 0.0, name='observation')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._state = 0
        self._episode_ended = False
        self._episode_reward = 0
        self.racer = tracks.Racer()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        self.racer.reset()
        return ts.restart(np.array([self._state], dtype=np.int32))


    def _step(self, action):
        if self._episode_ended:
            return self._reset()
         
        self.state, reward, done = self.racer.step(action)
        self._episode_reward += reward

        if done == True :
            self._episode_ended = True  
            reward = self._episode_reward
            self._episode_reward = 0
            print("Episode Ended\n")  
            return ts.termination(np.array([0], dtype=np.int32), reward)
        else:
            print("- action: ", action, "\treward: ", reward, "\tdone: ", done)
            return ts.transition(np.array([self._state], dtype=np.int32), reward=reward, discount=1)
        


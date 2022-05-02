from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from causal_world.actors.grasping_policy import GraspingPolicy
from causal_world.task_generators.towers import TowersGeneratorTask
import numpy as np

from causal_world.actors.base_policy import BaseActorPolicy
import os
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines import PPO2
except ImportError:
    pass

class MyTowerTask(TowersGeneratorTask):

    def __init__(self, **kwargs):
        super().__init__(variables_space='space_a_b',
                         activate_sparse_reward="False",
                         fractional_reward_weight=1,
                         tower_dims=[0.035, 0.035, 0.175],
                         tool_block_mass=0.08 ,
                         number_of_blocks_in_tower=np.array([1, 1, 5]))
        self._task_robot_observation_keys = [
            "time_left_for_task", "joint_positions", "joint_velocities",
            "end_effector_positions"
        ]



class MyTowerPolicy(BaseActorPolicy):

    def __init__(self):

        super(MyTowerPolicy, self).__init__('tower_policy')
        file="C:/Users/klakk/Documents/Me/ResearchPhD/CausalRobot/Tasks/tower_curr0/saved_model.zip"
        self._policy = PPO2.load(file)
        return

    def act(self, obs):

        return self._policy.predict(obs, deterministic=True)[0]



def solve_tower():
    task = MyTowerTask()
    # task = generate_task(task_generator_id='towers')
    env = CausalWorld(task=task, enable_visualization=True,
                      action_mode='end_effector_positions')
    # policy = GraspingPolicy(tool_blocks_order=[0, 1])
    policy = MyTowerPolicy()

    for _ in range(20):
        policy.reset()
        obs = env.reset()
        for _ in range(6000):
            obs, reward, done, info = env.step(policy.act(obs))
            # print("Reward is: ", reward)
    env.close()


if __name__ == '__main__':
    solve_tower()

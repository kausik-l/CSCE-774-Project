from causal_world.actors.base_policy import BaseActorPolicy
from causal_world.task_generators import generate_task
from causal_world.task_generators.stacked_blocks import StackedBlocksGeneratorTask
from causal_world.envs import CausalWorld
import os
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines import PPO2
except ImportError:
    pass

class MyStackingTask(StackedBlocksGeneratorTask):

    def __init__(self, **kwargs):
        super().__init__(variables_space='space_a_b',
                         num_of_levels=4,
                         tool_block_mass=0.08)
        self._task_robot_observation_keys = [
            "time_left_for_task", "joint_positions", "joint_velocities",
            "end_effector_positions"
        ]



class MyStackPolicy(BaseActorPolicy):

    def __init__(self):

        super(MyStackPolicy, self).__init__('stack_policy')
        file="../../trained_models/stacking2_curr0/model.zip"
        self._policy = PPO2.load(file)
        return

    def act(self, obs):

        return self._policy.predict(obs, deterministic=True)[0]



def test():
    task = MyStackingTask()
    env = CausalWorld(task=task, enable_visualization=True,
                      action_mode='end_effector_positions')
    # policy = GraspingPolicy(tool_blocks_order=[0, 1])
    policy = MyStackPolicy()

    for _ in range(20):
        policy.reset()
        obs = env.reset()
        for _ in range(6000):
            obs, reward, done, info = env.step(policy.act(obs))
            # print("Reward is: ", reward)
    env.close()


if __name__ == '__main__':
    test()

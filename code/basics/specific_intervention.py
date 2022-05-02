from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np


#change floor fiction
def friction_intervention():
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention({'floor_friction': 0.5})
        print("Intervention success signal", success_signal)
    env.close()


if __name__ == '__main__':
    friction_intervention()

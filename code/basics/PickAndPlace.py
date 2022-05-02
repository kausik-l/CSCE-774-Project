from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from causal_world.actors import PickAndPlaceActorPolicy



def example():
    task = generate_task(task_generator_id='pick_and_place')
    env = CausalWorld(task=task, skip_frame=3,
                      enable_visualization=True)
    policy = PickAndPlaceActorPolicy()
    env.close()


if __name__ == '__main__':
    example()

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.intervention_actors import GoalInterventionActorPolicy, VisualInterventionActorPolicy, \
    RandomInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper


def example():
    #initialize env
    task_gen = generate_task(task_generator_id='pushing')
    env = CausalWorld(task_gen, skip_frame=10, enable_visualization=True)

    # define a custom curriculum of interventions:
    env = CurriculumWrapper(env,
                            intervention_actors=[
                                # GoalInterventionActorPolicy(),
                                VisualInterventionActorPolicy(),
                                RandomInterventionActorPolicy(),
                                GoalInterventionActorPolicy()
                            ],
                            actives=[(2, 10, 1, 0), (20, 25, 1, 0), (25, 30, 1, 50)])

    for reset_idx in range(30):
        obs = env.reset()
        for time in range(100):
            print(time)
            desired_action = env.action_space.sample()
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    example()

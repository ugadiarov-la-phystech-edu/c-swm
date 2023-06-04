from gym.envs.registration import register


register(
    'ShapesChannelWiseBoxes6Act1Width8OneStaticGoalTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_active_boxes': 1,
        'n_passive_boxes': 4,
        'n_goals': 1,
        'width': 8,
        'render_scale': 10,
        'max_episode_steps': 100,
        'return_state': True,
        'channel_wise': True,
        'do_reward_push_only': False,
        'do_reward_active_box': True,
        'ternary_interactions': True,
    },
)


register(
    'ShapesChannelWiseBoxes6Act1Width8OneStaticGoalEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_active_boxes': 1,
        'n_passive_boxes': 4,
        'n_goals': 1,
        'width': 8,
        'render_scale': 10,
        'max_episode_steps': 100,
        'return_state': True,
        'channel_wise': True,
        'do_reward_push_only': False,
        'do_reward_active_box': True,
        'ternary_interactions': True,
    },
)


register(
    'ShapesChannelWiseBoxes6Act2Width8OneStaticGoalTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_active_boxes': 2,
        'n_passive_boxes': 3,
        'n_goals': 1,
        'width': 8,
        'render_scale': 10,
        'max_episode_steps': 100,
        'return_state': True,
        'channel_wise': True,
        'do_reward_push_only': False,
        'do_reward_active_box': True,
        'ternary_interactions': True,
    },
)


register(
    'ShapesChannelWiseBoxes6Act2Width8OneStaticGoalEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_active_boxes': 2,
        'n_passive_boxes': 3,
        'n_goals': 1,
        'width': 8,
        'render_scale': 10,
        'max_episode_steps': 100,
        'return_state': True,
        'channel_wise': True,
        'do_reward_push_only': False,
        'do_reward_active_box': True,
        'ternary_interactions': True,
    },
)


register(
    'ShapesChannelWiseBoxes6Act3Width8OneStaticGoalTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_active_boxes': 3,
        'n_passive_boxes': 2,
        'n_goals': 1,
        'width': 8,
        'render_scale': 10,
        'max_episode_steps': 100,
        'return_state': True,
        'channel_wise': True,
        'do_reward_push_only': False,
        'do_reward_active_box': True,
        'ternary_interactions': True,
    },
)


register(
    'ShapesChannelWiseBoxes6Act3Width8OneStaticGoalEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_active_boxes': 3,
        'n_passive_boxes': 2,
        'n_goals': 1,
        'width': 8,
        'render_scale': 10,
        'max_episode_steps': 100,
        'return_state': True,
        'channel_wise': True,
        'do_reward_push_only': False,
        'do_reward_active_box': True,
        'ternary_interactions': True,
    },
)

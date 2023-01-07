from gym.envs.registration import register

register(
    'ShapesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': True},
)

register(
    'ShapesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': True}
)

register(
    'ShapesTwoObjectsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': True},
)

register(
    'ShapesTwoObjectsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': True}
)

register(
    'ShapesRandomActionsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': True, 'random_actions': True},
)

register(
    'ShapesRandomActionsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': True, 'random_actions': True},
)

register(
    'ShapesNoHardWallsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': False}
)

register(
    'ShapesNoHardWallsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'hard_walls': False}
)

register(
    'ShapesStaticObjectTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 0, 'hard_walls': True}
)

register(
    'ShapesStaticObjectEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 0, 'hard_walls': True}
)

register(
    'ShapesStaticGoalTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'hard_walls': True}
)

register(
    'ShapesStaticGoalEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'hard_walls': True}
)

register(
    'ShapesStaticGoalNoHardWallsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'hard_walls': False}
)

register(
    'ShapesStaticGoalNoHardWallsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'hard_walls': False}
)

register(
    'CubesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)

register(
    'PushShapesHardWallsNoGoalsTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesHardWallsNoGoalsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesNoGoalsTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': False, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesNoGoalsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': False, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesHardWallsNoGoalsStaticBoxesTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 0, 'n_static_boxes': 1},
)

register(
    'PushShapesHardWallsNoGoalsStaticBoxesEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 0, 'n_static_boxes': 1},
)

register(
    'PushShapesHardWallsTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 4, 'n_goals': 1, 'n_static_boxes': 0},
)

register(
    'PushShapesHardWallsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 4, 'n_goals': 1, 'n_static_boxes': 0},
)

register(
    'PushShapesTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': False, 'n_boxes': 4, 'n_goals': 1, 'n_static_boxes': 0},
)

register(
    'PushShapesEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': False, 'n_boxes': 4, 'n_goals': 1, 'n_static_boxes': 0},
)

register(
    'PushShapesMovableGoalHardWallsTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0, 'static_goals': False},
)

register(
    'PushShapesMovableGoalHardWallsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0, 'static_goals': False},
)

register(
    'PushShapesMovableGoalHardWallsSmallTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0,
            'static_goals': False, 'width': 5, 'render_scale': 10},
)

register(
    'PushShapesMovableGoalHardWallsSmallEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'hard_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0,
            'static_goals': False, 'width': 5, 'render_scale': 10},
)

register(
    'ShapesNoWallsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'hard_walls': False},
)

register(
    'ShapesNoWallsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'hard_walls': False},
)

register(
    'WeightedShapesTrain-v0',
    entry_point='envs.weighted_block_pushing_rl:BlockPushingRL',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'observation_full_state': True, 'channels_first': True},
)

register(
    'WeightedShapesEval-v0',
    entry_point='envs.weighted_block_pushing_rl:BlockPushingRL',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'observation_full_state': True, 'channels_first': True},
)

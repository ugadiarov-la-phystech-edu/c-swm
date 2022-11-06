from gym.envs.registration import register

register(
    'ShapesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
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
    'PushSquaresTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'squares', 'n_boxes': 4, 'n_goals': 1},
)

register(
    'PushSquaresEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'squares', 'n_boxes': 4, 'n_goals': 1},
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

register(
    'ShapesStaticTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'hard_walls': True, 'num_movable_objects': 3},
)

register(
    'ShapesStaticEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'hard_walls':  True, 'num_movable_objects': 3},
)

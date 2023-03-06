from gym.envs.registration import register

register(
    'ShapesTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes5Width6Train-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 6,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes5Width6Eval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 6,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes5Width7Train-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 7,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes5Width7Eval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 7,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes4Width5Train-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 4,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes4Width5Eval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 4,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes3Width5Train-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 3,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesBoxes3Width5Eval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 3,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
    },
)

register(
    'ShapesChannelWiseTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': True,
    },
)

register(
    'ShapesChannelWiseEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 0,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': True,
    },
)

register(
    'ShapesTwoObjectsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': True},
)

register(
    'ShapesTwoObjectsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': True}
)

register(
    'ShapesTwoObjects3x3Train-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'width': 3, 'height': 3, 'num_objects': 2, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': True},
)

register(
    'ShapesTwoObjects3x3Eval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'width': 3, 'height': 3, 'num_objects': 2, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': True}
)

register(
    'ShapesRandomActionsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': True, 'random_actions': True},
)

register(
    'ShapesRandomActionsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': True, 'random_actions': True},
)

register(
    'ShapesNoHardWallsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': False}
)

register(
    'ShapesNoHardWallsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 0, 'num_goals': 0, 'border_walls': False}
)

register(
    'ShapesStaticObjectTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 0, 'border_walls': True}
)

register(
    'ShapesStaticObjectEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 0, 'border_walls': True}
)

register(
    'ShapesStaticGoalTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'border_walls': True}
)

register(
    'ShapesStaticGoalEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'border_walls': True}
)

register(
    'ShapesStaticGoalNoHardWallsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'border_walls': False}
)

register(
    'ShapesStaticGoalNoHardWallsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 5, 'num_static_objects': 1, 'num_goals': 1, 'border_walls': False}
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
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesHardWallsNoGoalsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesNoGoalsTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': False, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesNoGoalsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': False, 'n_boxes': 5, 'n_goals': 0},
)

register(
    'PushShapesHardWallsNoGoalsStaticBoxesTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 0, 'n_static_boxes': 1},
)

register(
    'PushShapesHardWallsNoGoalsStaticBoxesEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 0, 'n_static_boxes': 1},
)

register(
    'ShapesOneStaticGoalTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': False,
    },
)

register(
    'ShapesOneStaticGoalEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': False,
    },
)

register(
    'ShapesChannelWiseOneStaticGoalTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': True,
    },
)

register(
    'ShapesChannelWiseOneStaticGoalEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': True,
    },
)

register(
    'ShapesChannelWiseOneStaticGoalWithoutChannelTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': True,
        'channels_for_static_objects': False,
    },
)

register(
    'ShapesChannelWiseOneStaticGoalWithoutChannelEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'max_episode_steps': 100,
        'channel_wise': True,
        'channels_for_static_objects': False,
    },
)

register(
    'PushShapesTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': False, 'n_boxes': 4, 'n_goals': 1, 'n_static_boxes': 0},
)

register(
    'PushShapesEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': False, 'n_boxes': 4, 'n_goals': 1, 'n_static_boxes': 0},
)

register(
    'PushShapesMovableGoalHardWallsTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0, 'static_goals': False},
)

register(
    'PushShapesMovableGoalHardWallsEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0, 'static_goals': False},
)

register(
    'PushShapesMovableGoalHardWallsSmallTrain-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0,
            'static_goals': False, 'width': 5, 'render_scale': 10},
)

register(
    'PushShapesMovableGoalHardWallsSmallEval-v0',
    entry_point='envs.push:Push',
    max_episode_steps=75,
    kwargs={'observation_type': 'shapes', 'border_walls': True, 'n_boxes': 5, 'n_goals': 1, 'n_static_boxes': 0,
            'static_goals': False, 'width': 5, 'render_scale': 10},
)

register(
    'ShapesNoWallsTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'border_walls': False},
)

register(
    'ShapesNoWallsEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'border_walls': False},
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

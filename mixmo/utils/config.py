"""
General Variable definitions, mostly for testing purposes
"""

from easydict import EasyDict

cfg = EasyDict()

cfg.DEBUG = 0
cfg.SAVE_EVERY_X_EPOCH = 30
cfg.RATIO_EPOCH_DECREASE = 11/12

# TEST CONFIGS
cfg.CALIBRATION = EasyDict()
cfg.CALIBRATION.LRS = [0.001, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
cfg.CALIBRATION.MAX_ITERS = [5000]

cfg.RANDOM = EasyDict()
cfg.RANDOM.SEED = 1234
cfg.RANDOM.MAX_RANDOM = 10_000_000
cfg.RANDOM.SEED_OFFSET_MIXMO = 11
cfg.RANDOM.SEED_DA = 21
cfg.RANDOM.SEED_TESTVAL = 31

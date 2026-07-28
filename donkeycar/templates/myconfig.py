# """ 
# My CAR CONFIG 

# This file is read by your car application's manage.py script to change the car
# performance

# If desired, all config overrides can be specified here.
# The update operation will not touch this file.
# """

# Example: enable the "all_conditions" lighting-robustness augmentation
# profile (recommended when training one model on combined day/midday/
# night data). Uncomment to use; see cfg_complete.py for what each
# AUG_* setting does and other example profiles (indoor/outdoor/low_light).
# AUGMENTATIONS = ['BRIGHTNESS', 'BLUR', 'SHADOW', 'GAMMA', 'NOISE']
# AUG_GAMMA_RANGE = (60, 160)
# AUG_NOISE_PROBABILITY = 0.3

# Alternative: "contrast_focus" profile - leans further into color-
# invariance by regularly dropping color (GRAYSCALE) and/or de-emphasizing
# flat brightness in favor of edges/structure (HIGHPASS), on top of the
# same lighting-change augmentations. Try this if all_conditions still
# seems to rely too much on color-specific cues. Note both GRAYSCALE and
# HIGHPASS only run during training (see cfg_complete.py for the
# TRANSFORMATIONS alternative if you want the live camera feed to also
# always be grayscale/edge-only).
# AUGMENTATIONS = ['GRAYSCALE', 'HIGHPASS', 'GAMMA', 'NOISE', 'SHADOW']
# AUG_GAMMA_RANGE = (60, 160)
# AUG_GRAYSCALE_PROBABILITY = 0.5
# AUG_HIGHPASS_PROBABILITY = 0.5


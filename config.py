import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = '/home/jamesp/data/vcr/vcr1images'# os.environ['VCR_PARENT_DIR']
if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")

VCR_FEATURES_DIR = '/home/jamesp/data/visualcomet/features'
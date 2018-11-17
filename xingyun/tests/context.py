import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import attr_process, feature_process, img_process
import train, predict, evaluate
import vgg19
import utils
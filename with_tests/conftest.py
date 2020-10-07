import os
from os.path import abspath
from os.path import dirname as d

root_dir = d(d(abspath(__file__)))

os.environ['PROJECT_DIR'] = root_dir
testpaths = f"{root_dir}/with_tests"

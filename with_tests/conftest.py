import os
from os.path import dirname as d
from os.path import abspath, join

root_dir = d(d(abspath(__file__)))

os.environ['PROJECT_DIR'] = root_dir
testpaths = f"{root_dir}/with_tests"

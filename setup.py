#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script for avocadet catkin package."""

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['avocadet'],
    package_dir={'': 'src'},
    requires=['rospy', 'std_msgs', 'sensor_msgs', 'cv_bridge']
)

setup(**setup_args)

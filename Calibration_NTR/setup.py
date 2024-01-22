# -*- coding: utf-8 -*-
import logging
import os
import sys
from io import open
from os import path
import cal

try:
    from setuptools import setup, find_packages
except ImportError:
    logging.exception('Please install or upgrade setuptools or pip')
    sys.exit(1)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().split("\n")

package_data = []
base = path.join (here, 'cal')
for p,d,f in os.walk(base):
    if f: package_data.append (path.join (p[len(base)+1:], '*'))  # +1 to remove the / between cal and next dir

setup(
    name='cal',
    version=cal.__version__,
    description='A package with various CGI calibration tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/Calibration',
    author='AJ Riggs, David Marx, Eric Cady, Kevin Ludwick, Sam Halverson',
    author_email='aj.riggs@jpl.nasa.gov, david.s.marx@jpl.nasa.gov, eric.j.cady@jpl.nasa.gov, kjl0025@uah.edu, samuel.halverson@jpl.nasa.gov',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    package_data={
        'cal':package_data,
    },
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements
)

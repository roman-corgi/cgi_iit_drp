# -*- coding: utf-8 -*-
import logging
import sys
from io import open
from os import path

import proc_cgi_frame

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

setup(
    name='proc_cgi_frame',
    version=proc_cgi_frame.__version__,
    description='A package for processing CGI EMCCD Detector images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/proc_cgi_frame',
    author='Bijan Nemati, Sam Miller, Kevin Ludwick, Eric Cady',
    author_email='bijan@tellus1.com, sam.miller@uah.edu, kjl0025@uah.edu, eric.j.cady@jpl.nasa.gov',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=find_packages(),

    package_data={
        '': ['*'],
        'proc_cgi_frame':['testdata/*'],
    },

    python_requires='>=3.6',
    install_requires=requirements
)

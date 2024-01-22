# -*- coding: utf-8 -*-
import logging
import sys
from io import open
from os import path

try:
    from setuptools import setup, find_packages
except ImportError:
    logging.exception('Please install or upgrade setuptools or pip')
    sys.exit(1)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gsw_testing',
    version='1.4',  # Remember to update in __init__.py as well!
    description='A package with tools for testing of CTC GSW VIs.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/gsw_testing',
    author='A.J. Riggs',
    author_email='aj.riggs@jpl.nasa.gov',
    classifiers=[
        'Programming Language :: Python :: 3.10',
    ],
    packages=find_packages(),
    package_data={
        '': [
            'gsw_testing/*',
        ]
    },
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
    ]
)

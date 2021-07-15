#!/usr/bin/env python
"""The setup script."""

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=6.0', 'dbispipeline']

setup(
    author='Michael Voetter',
    author_email='michael.voetter@uibk.ac.at',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description='MediaEval2021 contains DBIS pipeline configurations.',
    entry_points={},
    install_requires=requirements,
    license='BSD license',
    long_description=readme,
    include_package_data=True,
    keywords='mediaeval2021',
    name='mediaeval2021',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    setup_requires=[],
    test_suite='tests',
    tests_require=[],
    url='https://dbis.uibk.ac.at',
    version='0.1.0',
    zip_safe=False,
)

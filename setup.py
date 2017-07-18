"""
Setup for Sairen.
"""

from setuptools import setup
import os.path
import re

HERE = os.path.abspath(os.path.dirname(__file__))
long_description = open(os.path.join(HERE, 'README.rst')).read()


def find_version(*file_paths):
    """:Return: the __version__ string from the path components `file_paths`."""
    with open(os.path.join(os.path.dirname(__file__), *file_paths)) as verfile:
        file_contents = verfile.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", file_contents, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='Sairen',
    version=find_version('sairen', 'env.py'),
    description='OpenAI Gym Reinforcement Learning Environment for the Stock Market',
    long_description=long_description,
    url='https://gitlab.com/doctorj/sairen',
    author='Doctor J',
    license='LGPL-3.0+',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='reinforcement learning openai gym finance trading interactive brokers',
    py_modules=['sairen'],
    install_requires=['ibroke', 'numpy', 'gym'],
    extras_require={'examples': ['h5py', 'keras', 'keras-rl', 'tensorflow', 'theano'], 'dev': ['sphinx']},
    package_data={},
)

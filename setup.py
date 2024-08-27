#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "pyoe"
DESCRIPTION = "Investigating open environment challenges in real-world relational data streams with PyOE."
URL = "https://github.com/sjtudyq/PyOE"
EMAIL = "liao_chengfan@sjtu.edu.cn"
AUTHOR = "Liao Chengfan"
REQUIRES_PYTHON = ">=3.10"
VERSION = "0.1.2"

# What packages are required for this module to be executed?
REQUIRED = [
    "autogluon.timeseries[chronos-openvino]==1.1.1",
    "catboost==1.2.5",
    "click==8.1.7",
    "combo==0.1.3",
    "copulas==0.11.0",
    "cvxopt==1.3.2",
    "delu==0.0.25",
    "einops==0.8.0",
    "keras==2.15.0",
    "lightgbm==4.3.0",
    "matplotlib==3.9.2",
    "menelaus==0.2.0",
    "networkx==3.3",
    "numpy==1.23.5",
    "pandas==2.2.2",
    "Pillow==10.4.0",
    "pyod==1.1.3",
    "pytorch_tabnet==4.1.0",
    "scikit_learn==1.4.0",
    "scikit_multiflow==0.5.3",
    "scipy==1.12.0",
    "seaborn==0.13.2",
    "tensorflow==2.15.0",
    "torch==2.3.1",
    "torchvision==0.18.1",
    "tqdm==4.66.4",
    "wget==3.2",
    "xgboost==2.0.3",
]

# What packages are optional?
EXTRAS = {
    "required": ["river==0.21.2", "rtdl==0.0.13", "streamad==0.3.1"],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)

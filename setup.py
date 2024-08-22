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
VERSION = "0.1.1"

# What packages are required for this module to be executed?
REQUIRED = [
    "absl-py==2.1.0",
    "accumulation_tree==0.6.2",
    "astunparse==1.6.3",
    "cachetools==5.4.0",
    "catboost==1.2.5",
    "certifi==2024.7.4",
    "charset-normalizer==3.3.2",
    "combo==0.1.3",
    "contourpy==1.2.1",
    "copulas==0.11.0",
    "coverage==7.6.0",
    "cycler==0.12.1",
    "delu==0.0.23",
    "einops==0.8.0",
    "et-xmlfile==1.1.0",
    "exceptiongroup==1.2.2",
    "fast-histogram==0.11",
    "flatbuffers==24.3.25",
    "fonttools==4.53.1",
    "gast==0.6.0",
    "google-auth==2.32.0",
    "google-auth-oauthlib==1.2.1",
    "google-pasta==0.2.0",
    "graphviz==0.20.3",
    "grpcio==1.65.1",
    "h5py==3.11.0",
    "idna==3.7",
    "iniconfig==2.0.0",
    "joblib==1.4.2",
    "keras==2.15.0",
    "kiwisolver==1.4.5",
    "libclang==18.1.1",
    "lightgbm==4.4.0",
    "llvmlite==0.43.0",
    "Markdown==3.6",
    "MarkupSafe==2.1.5",
    "matplotlib==3.9.1",
    "menelaus==0.2.0",
    "ml-dtypes==0.2.0",
    "mmh3==3.1.0",
    "numba==0.60.0",
    "numpy==1.23.5",
    "nvidia-cublas-cu11==11.10.3.66",
    "nvidia-cuda-nvrtc-cu11==11.7.99",
    "nvidia-cuda-runtime-cu11==11.7.99",
    "nvidia-cudnn-cu11==8.5.0.96",
    "nvidia-nccl-cu12==2.22.3",
    "oauthlib==3.2.2",
    "openpyxl==3.1.5",
    "opt-einsum==3.3.0",
    "packaging==24.1",
    "pandas==2.2.2",
    "patsy==0.5.6",
    "pillow==10.4.0",
    "plotly==5.23.0",
    "pluggy==1.5.0",
    "protobuf==4.25.3",
    "pyasn1==0.6.0",
    "pyasn1_modules==0.4.0",
    "pyod==1.1.3",
    "pyparsing==3.1.2",
    "pytest==7.4.4",
    "pytest-cov==4.1.0",
    "python-dateutil==2.9.0.post0",
    "pytorch-tabnet==4.1.0",
    "pytz==2024.1",
    "pyudorandom==1.0.0",
    "requests==2.32.3",
    "requests-oauthlib==2.0.0",
    "river==0.21.2",
    "rrcf==0.4.4",
    "rsa==4.9",
    "rtdl==0.0.13",
    "scikit-learn==1.5.1",
    "scikit-multiflow==0.5.3",
    "scipy==1.14.0",
    "seaborn==0.13.2",
    "six==1.16.0",
    "sortedcontainers==2.4.0",
    "statsmodels==0.13.5",
    "tdigest==0.5.2.2",
    "tenacity==8.5.0",
    "tensorboard==2.15.2",
    "tensorboard-data-server==0.7.2",
    "tensorflow==2.15.0",
    "tensorflow-estimator==2.15.0",
    "tensorflow-io-gcs-filesystem==0.37.1",
    "termcolor==2.4.0",
    "threadpoolctl==3.5.0",
    "tomli==2.0.1",
    "torch==1.13.1",
    "torchvision==0.14.1",
    "tqdm==4.66.4",
    "typing_extensions==4.12.2",
    "tzdata==2024.1",
    "urllib3==2.2.2",
    "Werkzeug==3.0.3",
    "wget==3.2",
    "wheel==0.43.0",
    "wrapt==1.14.1",
    "xgboost==2.1.0",
]

# What packages are optional?
EXTRAS = {
    "streamad": ["streamad==0.3.1"],
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

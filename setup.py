#! /usr/bin/env python
# License: 3-clause BSD
from setuptools import setup
import builtins


# This is a bit (!) hackish: we are setting a global variable so that the
# main modelcard __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components.
builtins.__MODELCARD_SETUP__ = True


import modelcard # noqa
import modelcard._min_dependencies as min_deps  # noqa

VERSION = modelcard.__version__

DISTNAME = "model-card"
DESCRIPTION = "A set of tools to create a model card"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Adrin Jalali"
MAINTAINER_EMAIL = "adrin.jalali@gmail.com"
URL = "http://github.com/model-card/model-card"
DOWNLOAD_URL = "https://pypi.org/project/model-card/#files"
LICENSE = "new BSD"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/model-card/model-card/issues",
    "Documentation": "https://github.com/model-card/model-card",
    "Source Code": "https://github.com/model-card/model-card",
}


def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Alpha",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            ("Programming Language :: Python :: " "Implementation :: CPython"),
        ],
        python_requires=">=3.6",
        install_requires=min_deps.tag_to_packages["install"],
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()

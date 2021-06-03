import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setuptools.setup(
    name="polarcycle",
    version="1.0.0",
    install_requires=install_requires,
    author="Cyprien Ruffino",
    author_email="ruffino.cyprien@protonmail.com",
    description="Extended CycleGAN for RGB-to-polarimetric image transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyprienruffino/PolarCycle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.5',
)
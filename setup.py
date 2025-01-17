from setuptools import find_packages, setup

with open("requirements.txt") as f:
    reqs = f.readlines()

setup(
    name="territorytools",
    version="1.0.0",
    description="Automated analysis of territorial behaviors including urine marking",
    url="https://github.com/FalknerLab/TerritoryTools.git",
    author="David Allen",
    author_email="da9769@princeton.edu",
    packages=['territorytools'],
    python_requires=">=3.8",
    install_requires=reqs,
    entry_points={"console_scripts": ["ptect=territorytools.cli:main",
                  "ptect-demo=tests.run_demo:main"]},
    license_files=("LICENCE",),
    license="BSD-3 Licence")

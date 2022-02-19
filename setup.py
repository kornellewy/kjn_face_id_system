from importlib.metadata import entry_points
from setuptools import find_packages, setup

# pip install pip -e .
setup(
    name="kjn_face_id_system",
    packages=find_packages(exclude=("tests*",)),
    entry_points={"console_scripts": ["kjn_face_id_system=kjn_face_id_system.cli:app"]},
)

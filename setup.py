# setup.py
from setuptools import setup, find_packages

setup(
    name="midisym",
    version="0.1.0",
    description="A custom simple MIDI reader based on mido",
    author="Eunjin Choi",
    author_email="jech@kaist.ac.kr",
    url="https://github.com/jech2/midisym",
    packages=find_packages(),
    install_requires=[
        'mido',  # 의존성 라이브러리
    ],
)
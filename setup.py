# setup.py
from setuptools import setup, find_packages

# setup.py
from setuptools import setup, Extension
import sys
import numpy

# 확장 모듈 정의
ext_modules = [
    Extension(
        name="midisym.mymodule",  # 모듈 이름
        sources=["src/mymodule.c"],  # C 소스 파일
        include_dirs=[numpy.get_include()],  # NumPy 헤더 포함 (필요 시)
        extra_compile_args=[],  # 추가 컴파일 옵션
    ),
    Extension(
        name="midisym.mymodule2",
        sources=["src/mymodule2.cpp"],
        include_dirs=[numpy.get_include()],  # Include numpy headers
        language="c++",  # Specify the language
        extra_compile_args=[
            "-std=c++11"
        ],  # Additional compiler arguments, e.g., C++ standard
    ),
    Extension(
        name="midisym.csamplers",
        sources=["src/gmsamplersmodule.c"],
        extra_link_args=[],
        include_dirs=[numpy.get_include(), "include"],
    ),
]

setup(
    name="midisym",
    version="0.1.0",
    description="A custom simple MIDI reader based on mido",
    author="Eunjin Choi",
    author_email="jech@kaist.ac.kr",
    url="https://github.com/jech2/midisym",
    packages=find_packages(),
    install_requires=[
        "mido",  # 의존성 라이브러리
    ],
    ext_modules=ext_modules,  # 확장 모듈
)

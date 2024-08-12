import os
import re

from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join("_version.py")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    return data["__version__"]


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines



def main():
    setup(
        name="doc_parser",
        version=read_version(),
        author="Licheng Wang",
        author_email="244267620@qq.com",
        description="Document parsing tool for LLM training and Rag",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["pdf", "BLOOM", "Falcon", "LLM", "ChatGPT", "transformer", "pytorch", "deep learning"],
        license="MIT",
        url="https://github.com/feb-co/DocParser",
        package_dir={"": "api/functional"},
        packages=find_packages("functional"),
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        entry_points={
        "console_scripts": [
            "docparser-pdf = pdf_parser:main",
        ],
    },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()

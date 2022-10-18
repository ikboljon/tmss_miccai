import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchmtlr",
    version="0.0.1",
    author="Michal Kazmierski",
    author_email="",
    description="PyTorch implementation of multi-task logistic regression for survival prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkazmier/torchmtlr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

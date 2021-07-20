import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="visepoch",
    version="0.1.0",
    author="George K. Holt",
    description="A package for visualising EPOCH results",
    long_description=long_description,
    license="MIT",
    packages=["visepoch"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8"
    ]
)
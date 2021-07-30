import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt", "r") as f:
    install_requires = [line.strip('\n') for line in f.readlines()]

setuptools.setup(
    name="visepoch",
    version='0.2.1',
    author="George K. Holt",
    description="A package for visualising EPOCH results",
    long_description=long_description,
    license="MIT",
    packages=["visepoch"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8"
    ],
    install_requires=install_requires
)
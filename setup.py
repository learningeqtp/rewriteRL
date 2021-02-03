import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rewriterl",  # Replace with your own username
    version="0.0.1",
    author="anonymized",
    author_email="anonymized",
    description="Reinforcement Learning environment for arithmetic rewriting tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/learningeqtp/rewriteRL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pydot",
        "matplotlib",
        "imageio",
        "lark-parser @ https://github.com/learningeqtp/lark/archive/master.zip",
    ],
    include_package_data=True,
)

from setuptools import setup, find_packages


setup(
    name='yoyotest',
    version='0.0.1',
    packages=find_packages(include=['yoyotest', 'yoyotest.*']),
    python_requires='>=3.7',
    install_requires=[
        'flake8',
        'flake8-docstrings',
        'tqdm',
        'mlflow',
        'orion>=0.1.8',
        'pyyaml>=5.3',
        'pytest>=4.6',
        'sphinx',
        'sphinx-autoapi',
        'sphinx-rtd-theme',
        'sphinxcontrib-napoleon',
        'sphinxcontrib-katex',
        'recommonmark',
        'torch'],
    entry_points={
        'console_scripts': [
            'main=yoyotest.main:main'
        ],
    }
)

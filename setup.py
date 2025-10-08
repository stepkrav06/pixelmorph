from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pixelmorph',
    version='0.1.0',  
    author='Stepan Kravtsov', 
    description='A command-line tool to transform and animate image morphs using optimization algorithms',
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    url='https://github.com/stepkrav06/pixelmorph', 
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'pixelmorph = pixelmorph.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
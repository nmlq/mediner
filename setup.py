
import setuptools


def read_requirements(filename: str):
    with open(filename) as f:
        return [line.strip() for line in f]


setuptools.setup(
    name="mediner",
    version='0.0.2',
    description="Medical NER",
    long_description="Named Entity Recognition for the VA Hospital",
    author="Nathan McCoy",
    author_email="noreply@va.gov",
    maintainer="Nathan McCoy",
    maintainer_email="noreply@va.gov",
    python_requires='>=3.10, <3.11',
    install_requires=read_requirements('requirements.txt'),
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"mediner": ["*.json", "*.cfg"]},
    entry_points={
        "console_scripts": ['mediner=mediner.cli:main'],
    },
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Healthcare Industry',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities',
    ],
)

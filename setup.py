# this file would be responsible for creating machine learning application as a package 
# with the help of setup.py  we would be able to build the entire machine learning application as package and even deplay it in pypy

from setuptools import find_packages,setup # this will automatically find out the packages in the entire ml application
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='Anti money laundering',
version='0.0.1',
author='Ishank Mittal',
author_email='mittalishank@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)

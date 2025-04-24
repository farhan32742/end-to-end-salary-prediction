from setuptools import setup, find_packages
from typing import List
#hypen_e_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        #if hypen_e_dot in requirements:
           # requirements.remove(hypen_e_dot)
    return requirements



setup(
    name="salary_prediction",  # Replace with your project name
    version="0.1.1",  # Replace with your project's version
    author="Farhan fayaz",  # Replace with your name
    author_email="farhanfiaz79@gmail.com.com",  # Replace with your email
    packages=find_packages(),  # Automatically finds and includes packages
    install_requires=get_requirements("requirements_dev.txt")
)

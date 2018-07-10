# Stock Market Prediction Using Machine Learning

## Structure
The repository houses:
* 'datasets' folder that is populated with stock data the first time script is run. To repopulate data:
  ```bash
  python get_data.py
  ```
* 'research-papers' folder - the papers referred during the development of the model.

## Running for the first time?
*The files environment.yml, requirements.txt make it easy to replicate the environment required for running the model.*
### Setting up the Environment
1. For anaconda:<br>
   1. To install anaconda, [refer this](https://conda.io/docs/user-guide/install/index.html)<br>
   2. The base directory contains 'environment.yml' file. To replicate the same environment:
      ```bash
      conda env create -f environment.yml
      ```
2. For python3 virtual environment:<br>
   1. To install virtualenv, refer [this](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to)
      ```bash
      pip install virtualenv
      virtualenv --python=python3 ml-stock-prediction
      ```
   2. The base directory contains 'requirements.txt' file. To install the required packages:
      ```bash
      pip install -r requirements.txt
      ```

## To-do:
1. Complete data-preprocessing
2. Add functions to plot and analyse data before any feature-scaling
3. Implement functions for feature scaling.
4. Add features.
5. Make data ready for input to the model.
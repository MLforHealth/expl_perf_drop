from setuptools import setup, find_packages


setup(name='expl_perf_drop',
      version='0.4',
      description='Attributing performance drops to distribution shifts',
      url='https://github.com/MLforHealth/expl_perf_drop',
      author='Haoran Zhang',
      author_email='haoranz@mit.edu',
      license='BSD',
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires= [
        'numpy>=1.19.0',
        'pandas>=1.1.5',
        'scikit-learn>=0.24.1',
        'scipy>=1.5.1',
        'tqdm',
        'imbalanced-learn>=0.8.0',
        'xgboost>=0.90',
        'torch'
      ])
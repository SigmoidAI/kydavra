from setuptools import setup
long_description = '''
# kydavra
Kydavra is a python sci-kit learn inspired package for feature selection. It used some statistical methods to extract from pure pandas Data Frames the columns that are related to column that your model should predict.
This version of kydavra has the next methods of feature selection:
* ANOVA test selector (ANOVASelector)
* Chi squared selector (ChiSquaredSelector)
* Genetic Algorithm selector (GeneticAlgorithmSelector)
* Kendall Correlation selector (KendallCorrelationSelector)
* Lasso selector (LassoSelector)
* Pearson Correlation selector (PearsonCorrelationSelector)
* Point-Biserial selector (PointBiserialCorrSelector)
* P-value selector (PValueSelector)
* Spearman Correlation selector (SpermanCorrelationSelector)
* Shannon selector (ShannonSelector)
* ElasticNet Selector (ElasticNetSelector)
* M3U Selector (M3USelector)
* MUSE Selector (MUSESelector)
* Mixer Selector (MixerSelector)
* PCA Filter (PCAFilter)
* PCA Reducer (PCAReducer)
* LDA Reducer (LDAReducer)
* Bregman Divergence selector (BregmanDivergenceSelector)
* Fisher Selector (FisherSelector)
* ICA Reducer (ICAReducer)
* ICA Filter (ICAFilter)
* Itakura-Saito Divergence selector (ItakuraSaitoSelector)
* Jensen-Shannon Divergence selector (JensenShannonSelector)
* Kullback-Leibler selector (KullbackLeiblerSelector)
* MultiSURF selector (MultiSURFSelector)
* Phik selector (PhikSelector)
* ReliefF selector (ReliefFSelector)

All these methods takes the pandas Data Frame and y column to select from remained columns in the Data Frame.

How to use kydavra\
To use selector from kydavra you should just import the selector from kydavra in the following framework:
```python
from kydavra import PValueSelector
```
class names are written above in parantheses.\
Next create a object of this algorithm (I will use p-value method as an example).
```python
method = PValueSelector()
```
To get the best feature on the opinion of the method you should use the 'select' function, using as parameters the pandas Data Frame and the column that you want your model to predict.
```python
selected_columns = method.select(df, 'target')
```
Returned value is a list of columns selected by the algorithm.

Some methods could plot the process of selecting the best features.\
In these methods dotted are features that wasn't selected by the method.\
*ChiSquaredSelector*
```python
method.plot_chi2()
```
For ploting and
```python
method.plot_chi2(save=True, file_path='FILE/PATH.png')
```
and
```python
method.plot_p_value()
```
for ploting the p-values.\
*LassoSelector*
```python
method.plot_process()
```
also you can save the plot using the same parameters.\
*PValueSelector*
```
method.plot_process()
```

Some advice.
* Use ChiSquaredSelector for categorical features.
* Use LassoSelector and PValueSelector for regression problems.
* Use PointBiserialCorrSelector for binary classification problems.
* Use ShannonSelector to choose whatever to keep the NaN values (as another value) and to drop column with a lot of NaN values.\n

With love from Sigmoid.

We are open for feedback. Please send your impression to vladimir.stojoc@gmail.com
'''
setup(
  name = 'kydavra',
  packages = ['kydavra'],
  version = '0.3.4',
  license='MIT',
  description = 'Kydavra is a sci-kit learn inspired python library with feature selection methods for Data Science and Macine Learning Model development',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'SigmoidAI - Păpăluță Vasile, Stojoc Vladimir',
  author_email = 'vladimir.stojoc@gmail.com',
  url = 'https://github.com/SigmoidAI/kydavra',
  download_url = 'https://github.com/ScienceKot/kydavra/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ml', 'machine learning', 'feature selection', 'python'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scikit-learn',
          'statsmodels',
          'matplotlib',
          'seaborn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Framework :: Jupyter',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
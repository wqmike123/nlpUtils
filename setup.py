# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:30:58 2019

@author: wq
"""

import setuptools

setuptools.setup(
    name = "nlpUtils",
    version = "0.0.1",
    author = "wq",
    author_email = "wqmike123@gmail.com",
    description = "some tools for nlp tasks",
    url="https://github.com/wqmike123/nlpUtils",
    packages=setuptools.find_packages(),
    install_requires=[
                      'gensim>=3.6.0',
                      'pandas>=0.23.4',
                      'numpy>=1.16.1',
                      'tensorflow>=1.12.0',
                      'tensorflow_hub>=0.2.0',
                      'nltk>=3.2.3',
                      'wordcloud>=1.5.0',
                      'ciseau>=1.0.1',
                      'textblob>=0.15.2',
                      'spacy>=2.0.18',
                      'seaborn>=0.7.1',
                      'gensim>=3.6.0',
                      'sklearn'
                    ],

)

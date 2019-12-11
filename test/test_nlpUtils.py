# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:41:47 2019

@author: wq
"""

from nlpUtils import preprocessPipeline

test_text = """Market conditions were historically tough in the fourth quarter, 
            but we delivered a resilient overall performance thanks to the strength of 
            our strategic choices and diversified franchise. 
            """

res = preprocessPipeline(test_text, stem = True,stopword = True,lower = True, padding = 100,clean =True, tokenTool = 'unitok')
print(res)

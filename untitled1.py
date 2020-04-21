#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:04:44 2020

@author: roeepartoush
"""
# %%
from pymed import PubMed
import pandas as pd
import numpy as np

# Create a PubMed object that GraphQL can use to query
# Note that the parameters are not required but kindly requested by PubMed Central
# https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
pubmed = PubMed(tool="MyTool", email="my@email.address")

# Create a GraphQL query in plain text
query = "retina[Title]"

yr=np.zeros

# Execute the query against the API
N=20000
results = pubmed.query(query, max_results=N)
#results2 = pubmed.query(query, max_results=N)
#artsDF = pd.DataFrame(columns=['article','year'],index=np.arange(len(tuple(results2))))
#artsDF = pd.DataFrame(columns=['article','year'],index=np.arange(N))
# Loop over the retrieved articles
arts=list()

for article in results:
#    print('hi!')
#    artsDF.iloc[i]['article'] = article
#    artsDF.iloc[i]['year'] = article.publication_date.year
#    i=i+1
    arts.append(article)
    # Print the type of object we've found (can be either PubMedBookArticle or PubMedArticle)
#    print(type(article))
#
#    # Print a JSON representation of the object
#    print(article.toJSON())
# %%

def toYear(pubDate):
    if type(pubDate)==str:
        yr = int(pubDate)
    else:
        yr = pubDate.year
    return yr

artsDF = pd.DataFrame(columns=['article','year'],index=np.arange(len(arts)))
i=0
for i in np.arange(len(artsDF)):
#    print('hi!')
    artsDF.iloc[i]['article'] = arts[i]
    artsDF.iloc[i]['year'] = toYear(arts[i].publication_date)
    i=i+1
    

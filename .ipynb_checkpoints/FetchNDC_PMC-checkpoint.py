import requests
import re

def downloadFile(url, filename):
    r = requests.get(url, allow_redirects=True)
    print(filename)
    open(filename, 'wb').write(r.content)
    return filename

import zipfile
def unzip(filename, unzipdir):
    with zipfile.ZipFile(filename,"r") as zip_ref:
        zip_ref.extractall(unzipdir)
        
import pandas as pd

def readProduct(productfile):
    df = pd.read_csv(productfile, sep="\t", encoding='latin-1')
    return df


import codecs
import pandas as pd
import io
import random

def splitDataSet(df,fieldName):
    names = df[fieldName].sample(n=1000, random_state=1).unique()
    from sklearn.model_selection import train_test_split
    train_x, test_x, x, y = train_test_split((names),(names), test_size=0.2)
    print(f"{len(train_x)} train_x")
    print(f"{len(test_x)} test_x")
    return(train_x, test_x)


import json
import pandas as pd

def downloadCorpsFromPMC(keyword):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={keyword}&resultType=core&cursorMark=*&pageSize=200&format=json"
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    prod_dict = json.loads(response.content)
    result = []    
    if(prod_dict['hitCount'] > 0):
        for item in prod_dict['resultList']['result']: 
            if('abstractText' in item):
                if(keyword in item['abstractText']):
                    result.append(item['abstractText'])               
            if('title' in item):
                if(keyword in item['title']):
                    result.append(item['title'])                
    print(f"{len(result)} found")
    return result

def download(keywords, output_file):
    f= open(output_file,"w+")
    result = []
    for keyword in keywords:
        print(f" keyword {keyword} load...")
        texts = downloadCorpsFromPMC(keyword)
        if(len(texts) > 0):
            for text in texts: 
                f.write(json.dumps({"keyword":keyword, "texts": text}) + "\n")
    f.close()

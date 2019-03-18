import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import os

def buildCorp(rowIndex, count, keyword, text, label, pipeName, nlp, out):
    # use spacy entity ruler to generate the bootstrap corps
    ruler = EntityRuler(nlp)
    word_list = []
    for w in keyword.split(" "):
        word_list.append({"lower": w.lower()})
    patterns = [{"label": label, "pattern": word_list}]
    ruler.add_patterns(patterns)
    nlp.replace_pipe(pipeName, ruler)
    if(rowIndex % 100 ==0):
        print(f"{rowIndex}/{count}")
    doc = nlp(text)
    
    temp_index = 0
    for sent in doc.sents:
        isKeywordExist = keyword in sent.text
        if(isKeywordExist and len(sent) > 2):
            if(temp_index == 0):
                temp_index += 1
                out.write("-DOCSTART- -X- - O\n\n")
            for token in sent:
                # ignore 3rd argument as "-" since alot of library seems to ignore the chunk tag
                if(token.ent_type_):
                    # override the spacy's entity rule if it is found in label matches
                    if(token.text == keyword):
                        out.write(f"{token.orth_} {token.tag_} - {token.ent_iob_}-{label}\n")
                    else:
                        out.write(f"{token.orth_} {token.tag_} - {token.ent_iob_}-{token.ent_type_}\n")
                else:
                    out.write(f"{token.orth_} {token.pos_} - O\n")
            out.write("\n")
def build_conll_03(df, keyfield, textfield, label, outfile):
    print(f"writing {outfile} ...")
    ruler_name= 'custom'
    nlp = spacy.load('en')
    ruler = EntityRuler(nlp)
    nlp.add_pipe(ruler, name=ruler_name)
    out = open(outfile, "w")
    # write start tag for conll_03 format
    count = len(df)
    df.insert(0, 'row_num', range(0, len(df)))
    df.apply(lambda r: buildCorp(r['row_num'], count, r[keyfield], r[textfield], label, ruler_name, nlp, out), axis=1)
    out.close()
    return count

def combineFiles(filenames, outputfile):
    print(f"combining {filenames}")
    print(f"writing to {outputfile}")
    with open(outputfile, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                
def convert(input_f, output_f):
    out = open(output_f, "w")
    with open(input_f) as fp:  
        for cnt, line in enumerate(fp):
            values = line.split(" ")
            if(len(values) > 3):
                out.write(f"{values[0]} {values[1]} - {values[3]}")
            else:
                out.write(line)
    fp.close()
    out.close()
    
import pandas as pd
import numpy as np

from enum import Enum
class CorpsType(Enum):
    DEV = 1
    TEST = 2
    TRAIN = 3

    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def rmdir(path):
    if not os.path.exists(path):
        os.rmdir(path)
        
def convertJson2Conll_03(jsonfile, basedir, corpsname, label, max=1000):
    mkdir(basedir)
    mkdir (basedir + "/conll_03")
    outputname = basedir + '/conll_03/' + corpsname
    print(outputname)
    df = pd.read_json(jsonfile, lines=True)
    if(len(df) > max):
        df = df[:max]
        
    build_conll_03(df, 'keyword', 'texts', label, outputname)
    return outputname

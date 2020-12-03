'''

This is the main initialization file for Wesley. It will load functions and libraries needed in the main program. I have embedded library import commands within the functions. This adds to the processing time as these libraries will potentially be reloaded several times, but this allows the functions to operate independently as modules - should they be separated at any time from this initializing file. 


Coding by Ben P. Meredith, Ed.D. except where otherwise noted

'''

#-------------------------
''' 

Import first of the need libraries.
Some libraries are also imported within functions to make them more modular,
but this has its own problem with processing time.

'''

import os
import pandas as pd
import pdfminer
import re 
from pprint import pprint
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import shutil  
from nltk import tokenize
import textract
from gensim.summarization import keywords



#-----------------------------------
# Determine if a file exists within a pathway
def find_file(pathway):#pathway is the path to the file
    from pathlib import Path
    pathway = Path(pathway)# convert the pathway to an actual path from a string
    if pathway.exists():#Determine if the file exists
        return 1 #1 = file exists
    else: 
        return 0 # 0 = file does not exist

#----------------------------------
# Walk a directory

def walk_directory(directory):
    iterator = 0
    file_list = []

    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.startswith('.'):
                pass
            elif file.endswith('.csv'):
                pass
            else:
#                 df.loc[iterator, 'root'] = root
#                 df.loc[iterator, 'file'] = file
    #             print('len(files):', len(files))
    #             print('root:', root)
    #             print('dirs:', dirs)
    #             print('files:', files)
                pathway = os.path.join(root, file)
                pathway = os.path.realpath(pathway)# added from Jeanna for Windows to read
                file_list.append(pathway)
                iterator += 1
                
    return file_list

#----------------------------------
# Walk a directory and return list of file names

def walk_directory_verbose(directory):
    iterator = 0
    file_list = []

    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.startswith('.'):
                pass
            else:
                print('len(files):', len(files))
                print('root:', root)
                print('dirs:', dirs)
                print('files:', files)
                pathway = os.path.join(root, file)
                pathway = os.path.realpath(pathway)# added from Jeanna for Windows to read
                file_list.append(pathway)
                iterator += 1
                
    return file_list

#-------------------------
#FUNCTIONS follow below

#Initialize the data_log by discovering if it exists. If it does, load it. Otherwise, form one. 
def initialize_data_log():
    import pandas as pd
    answer = find_file('data_table.csv')
    if answer == 1:
        df = pd.read_csv('data_table.csv', index_col=0)
        df = df.drop(['level_0'], axis=1, errors='ignore')#Drops level_0 column that keeps showing up 
    else:
        df = pd.DataFrame(columns=('file_name', 'file_type', 'raw_text', 'clean_text', 'word_token', 'key_words', \
                                  'urls', 'emails'))
        df.to_csv('data_table.csv')
    return df, answer


#--------------------------------------------
#Initialize synonyms log by discovering if it exists. It it does, load it. Otherwise, form one.
def initialize_syn_log():
    import pandas as pd
    answer = find_file('syn_table.csv')
    if answer == 1:
        syndf = pd.read_csv('syn_table.csv', index_col=0)
        syndf = syndf.drop(['level_0'], axis=1, errors='ignore')#Drops level_0 column that keeps showing up 
    else:
        syndf = pd.DataFrame(columns=('word', 'synonyms'))
        syndf.to_csv('syn_table.csv')
    return syndf, answer

#--------------------------------------------
#Initialize url log by discovering if it exists. It it does, load it. Otherwise, form one.
def initialize_url_log():
    import pandas as pd
    answer = find_file('url_results.csv')
    if answer == 1:
        urldf = pd.read_csv('url_results.csv', index_col=0)
        urldf = urldf.drop(['level_0'], axis=1, errors='ignore')#Drops level_0 column that keeps showing up 
    else:
        urldf = pd.DataFrame(columns=('count', 'url'))
        urldf.to_csv('url_results.csv')
    return urldf, answer


#---------------------------------------------
#Pull text from a PDF given the document path (written by Jeanna Shoonmaker)
def pdf_to_txt(path):
    #from pdfminer.six documentation: https://pdfminersix.readthedocs.io/en/latest/tutorial/composable.html
    from io import StringIO
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser
    output_string = StringIO()
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    text = str(output_string.getvalue())
    return text
    
#--------------------------------------------
#Pull text from a TXT given the document path

def txt_to_txt(path):
    try:
        tempopen = open(path, 'rt', errors='ignore') # open the file for reading text
        temptext = tempopen.read()  # read the entire file as a string
        tempopen.close()            # close the file
        
    except TypeError:
        tempopen = open(path, 'rt', encode='utf-8') # open the file for reading text
        temptext = tempopen.read()  # read the entire file as a string
        tempopen.close()            # close the file
        
    return temptext                 # return the raw text

#------------------------------------------
#Pull text from .docx file given the document path

def docx_to_txt(path):
    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    PARA = WORD_NAMESPACE + 'p'
    TEXT = WORD_NAMESPACE + 't'

    
    try:
        from xml.etree.cElementTree import XML
    except ImportError:
        from xml.etree.ElementTree import XML
    import zipfile
    """
    Take the path of a docx file as argument, return the text in unicode.
    """
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = XML(xml_content)

    paragraphs = []
    for paragraph in tree.getiterator(PARA):
        texts = [node.text
                 for node in paragraph.getiterator(TEXT)
                 if node.text]
        if texts:
            paragraphs.append(''.join(texts))

    return '\n\n'.join(paragraphs)

#------------------------------------
# rtf to txt

def rtf_to_txt(path):
    from striprtf.striprft import rtf_to_text
    rtf = str(path)
    text = rtf_to_text(rtf)
    return text

#--------------------------------
# Bytes to Strings

def to_string(text):
    if isinstance(text, bytes):
        converted_text = text.decode('utf-8')
    else:
        converted_text = text
    return converted_text

#--------------------------------
# Strings to Bytes

def to_bytes(text):
    if isinstance(text, str):
        converted_text = text.encode('utf-8')
    else:
        converted_text = text
    return converted_text

#--------------------------------
# Pulled Text Cleaning Function

def clean_the_text(text, remove_numbers=False):
#     from bs4 import BeautifulSoup
#     soup = BeautifulSoup(text, 'lxml')
    
#     from pattern.web import URL, plaintext
#     text = plaintext(str(text), keep=[], linebreaks=2, indentation=False)

    import unicodedata
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii').decode('utf-8')
    
    import re
#     clean = re.compile(r'^<.*?>}{')
#     text = re.sub(clean, '', text)
    text = re.sub('x9307.', '', text)
    text = text.replace('\xef\xac\x81', 'fi')
    text = text.replace(' .org', '.org')
    text = text.replace('ISSN ', 'ISSN')
    text = text.replace('xef', '')
    text = text.replace('x83', '')
    text = text.replace('x98', '\n')
    text = text.replace('x99', "'")
    text = text.replace('xe2', ' ') 
    text = text.replace('x80', ' ')
    text = text.replace('xa2', '*')
    text = text.replace('\\xc3\\xa9', 'e')
    text = text.replace('\\xe2\\x80\\x90', ' - ')
    text = text.replace('\\xe2\\x80\\x91', '-')
    text = text.replace('\\xe2\\x80\\x92', '-')
    text = text.replace('\\xe2\\x80\\x93', '-')
    text = text.replace('\\xe2\\x80\\x94', '-')
    text = text.replace('\\xe2\\x80\\x98', "'")
    text = text.replace('\\xe2\\x80\\x99', "'")
    text = text.replace('\\xe2\\x80\\x9b', "'")
    text = text.replace('\\xe2\\x80\\x9c', '"')
    text = text.replace('\\xe2\\x80\\x9d', '"')
    text = text.replace('\\xe2\\x80\\x9e', '"')
    text = text.replace('\\xe2\\x80\\x9f', '"')
    text = text.replace('\\xe2\\x80\\xa6', '...')
    text = text.replace('\\xe2\\x81\\xba', "+")
    text = text.replace('\\xe2\\x81\\xbb', "-")
    text = text.replace('\\xe2\\x81\\xbc', "=")
    text = text.replace('\\xe2\\x81\\xbd', "(")
    text = text.replace('\\xe2\\x81\\xbe', ")")  
#     text = text.replace("\'", "'")
    text = text.replace('\\n', '\n ')
    text = text.replace('\n\n', '\n  ')
    text = text.replace('\\xc2\\xae', ' ')
#     text = text.replace('\n','    ') # new line
    text = text.replace('\t','     ')
    text = text.replace('\s', ' ') # space
    text = text.replace('\r\r\r', ' ')#carrage Return
    text = text.replace('\\xc2\\xa9 ', ' ')
    text = text.replace('xe2x80x93', ',')
    text = text.replace('xe2x88x92', ' ')
    text = text.replace('\\x0c', ' ')
    text = text.replace('\\xe2\\x80\\x9331', ' ')
    text = text.replace('xe2x80x94', ' ')
    text = text.replace('\x0c', ' ')
    text = text.replace(']', '] ')
#     text = text.replace(' x99', "'")
#     text = text.replace('xe2x80x99', "'")
    text = text.replace('\\xe2\\x80\\x933', '-')
    text = text.replace('\\xe2\\x80\\x935', '-')
    text = text.replace('\\xef\\x82\\xb7', ' ')
    text = text.replace('\\', ' ')
    text = text.replace('xe2x80x99', "'")
    text = text.replace('xe2x80x9cwexe2x80x9d', ' ')
    text = text.replace('xe2x80x93', ', ')
    text = text.replace('xe2x80x9cEUxe2x80x9d', ' ')
    text = text.replace('xe2x80x9cxe2x80x9d', ' ')
    text = text.replace('xe2x80x9cAvastxe2x80x9d', ' ')
    text = text.replace('xc2xa0', ' ')
    text = text.replace('xe2x80x9cxe2x80x9d', ' ')
    text = text.replace('xe2x80x9c', ' ')
    text = text.replace('xe2x80x9d', ' ')
    text = text.replace('xc2xad',' ')
    text = text.replace('x07', ' ')
    text = text.replace('tttttt', ' ')
    text = text.replace('activetttt.', ' ')    
    text = text.replace('.sdeUptttt..sdeTogglettttreturn', ' ') 
    text = text.replace('ttif', ' ')
    text = text.replace('.ttt.', ' ')
    text = text.replace(" t t ", ' ')
    text = text.replace('tttt ', ' ')
    text = text.replace(' tt ', ' ')
    text = text.replace(' t ', ' ')
    text = text.replace(' t tt t', ' ')
    text = text.replace('ttt', ' ')
    text = text.replace('ttr', ' ')
    text = text.replace('.display', ' ')
    text = text.replace('div class', ' ')
    text = text.replace('div id', ' ')
    text = text.replace('Pocy', 'Policy')
    text = text.replace('xc2xa0a', ' ')
    text = text.replace(' b ', '')
    text = text.replace('rrrr', '')
    text = text.replace('rtttr', '')
    text = text.replace('    ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace(' r ', ' ')
    text = text.replace(' tr ', ' ')
    text = text.replace(' rr  r  ', ' ')
    text = text.replace('   tt t t rt ', ' ')
    text = text.replace('r rrr r trr ', ' ')
    text = text.replace(' xe2x80x93 ', ' ')
    text = text.replace(' xe6xa8x82xe9xbdxa1xe6x9cx83  ', ' ')
    text = text.replace(' rrr ', ' ')
    text = text.replace(' rr ', ' ')
    text = text.replace('tr ', '')
    text = text.replace(' r ', '')
    text = text.replace("\'", "")
    text = text.replace(' t* ', ', ')
    text = text.replace('[pic]', '')
    text = text.replace('    ', '')
    text = text.replace('|', '')
    text = text.replace('__', '')
    text = text.replace('b"', '')
    text = text.replace('xe2x80xa2', '. ')
    text = text.replace('\x0c', '')
    text = text.replace('xc2', '')
    text = text.replace('xa0', '')
    text = text.replace('x99s', '- ')
    text = text.replace('x9d', '')
    text = text.replace('x9c', '')
    text = text.replace(' x93 ', ': ')
    text = text.replace('....', '')
    text = text.replace(' s ', "'s ")
    text = text.replace(' xac x81', ' fi')
    text = text.replace('peci fi', 'pecifi')
    text = text.replace('xc3 x9f', 'copyright')
    text = text.replace(' et al ', 'etal')
    text = text.replace('x97 x8f ', '*')
    text = text.replace(' x937 ', ': ')
    text = text.replace(' xac x82', 'sp') 
    text = text.replace(' x90', '-')
    text = text.replace('- ', '')
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
    
    return text

#---------------------------------
# Tag Text with Part of Sentence (POS) tags given the text as a string and returning the POS tags

def pos_tag(text):
    import nltk
    from nltk import pos_tag
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_w = word_tokenize(text)
    return pos_tag(token_text_w)# Normalize Corpus

#----------------------------------
#Count unique words in a document given the document text given as a string and returning the count

def count_unique_words(string):
    string_list = string.split(' ')
#     print(len(string_list))
    count = {}

    for i in string_list:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    return count

#---------------------------------
#Count TOTAL words in a document given the document text given as a string and returning the count as an integer

def count_words(text):
    count = 0
    for words in text.split(' '):
        count += 1
    return int(count)

#--------------------------------
# Count the FREQUENCY of words in a document text given as a string and returning the frequency

def count_word_frequency(text):

    import re
    import string

    frequency = {}

    text_string = str(text).lower()
    match_pattern = re.findall(r'\b[a-z]{1,15}\b', text_string)

    for word in match_pattern:
        count = frequency.get(word,0)
        frequency[word] = count + 1

    frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)

    return frequency

#----------------------------------
#Simple n-gramming function

def ngram_list(text, n_gram_size):
    from nltk import ngrams
    output = []
    grams = ngrams(text.split(), n_gram_size)
    for gram in grams:
        output.append(gram)
    return output


#----------------------------------
# Find Keywords in a document text given as a string and returning the keywords found
''' Note: This keyword pulling function is very basic and needs further refinement. In particular, its pulled words list criteria is unclear. For now, it works, but it will need to be refined to pull specific keywords based upon the word's POS.
'''

def find_keywords(text):
    key_words = keywords(text)
    return key_words

#---------------------------------
#Tokenize a document's text given as a string to the word level and returning a list of tokenized words

def tokenize_by_words(text):
    from nltk.tokenize import word_tokenize
    token_text_w = str(word_tokenize(text))
    return str(token_text_w)

def tokenize_by_words2(text):
    text = text.replace('\n', ' ')
    text = text.split(' ')
    return text
#---------------------------------
# Split a document's text given as a string into sentences

def split_into_sentences(text):
    import re
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

#----------------------------------------
# Simple sentence tokenizer given the document text as a string

def simple_tokenize_by_sentences(text):
    text = text.split('. ')
    return text

#--------------------------------------
#Tokenize by Paragraph given the document text as a string and returning a list of paragraphs

def tokenize_by_paragraphs(text):
    paragraphs = text.split('\n\n')
    return paragraphs

#-------------------------------------
#Remove Stopwords given the document text as a string and returning filtered text

def remove_stopwords(text, is_lower_case=False):#Lower Case only = False
    import nltk
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenize_by_words(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#------------------------------------
#Simple Stemmer given the document text as a string and returning the stemmed text
'''
Note: This is a very simple stemmer. It does not do the best job on all texts. A better stemmer should be explored. 
'''

def simple_stemmer(text):
    from nltk.stem import PorterStemmer
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

#-------------------------------------
#Simple Lemmatizer given the document text as a string and returning the stemmed text

def lemmatize_text(text):
    from nltk import nlp
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

#------------------------------------
#Single Word Search
def single_word_search(search_word):
    iterator = 0

    for article in df["clean_text"]:
        try:
            article = str(article).split('\n\n')
            for sentence in article:
                if search_word in sentence:
                    print((sentence.replace(search_word, '\033[43;30m{}\033[m'.format(search_word))), '\n\n')#highlight search term
        except TypeError:
            continue

        iterator += 1

#--------------------------------------
# Created, modified datetime

def accessed_created_modified(file):
    import os.path, time
    import datetime

    currenttime = (datetime.datetime.now())
    accessed = currenttime.strftime("%b %d, %Y, %H:%M")
    modified = time.ctime(os.path.getmtime(file))
    created = time.ctime(os.path.getctime(file))
    return accessed, created, modified

#---------------------------------------

# Code Snippet for Top N-grams Barchart
def plot_top_ngrams_barchart(text, n=2):
    import seaborn as sns
    import numpy as np
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    from collections import  Counter

    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:25]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x)
    
#---------------------------------------
# Named Entity Barchart
def plot_named_entity_barchart(text):
    import spacy
    from collections import  Counter
    import seaborn as sns

    nlp = spacy.load("en")
    
    def _get_ner(text):
        doc=nlp(text)
        return [X.label_ for X in doc.ents]
    
    ent=text.apply(lambda x : _get_ner(x))
    ent=[x for sub in ent for x in sub]
    counter=Counter(ent)
    count=counter.most_common()
    
    x,y=map(list,zip(*count))
    sns.barplot(x=y,y=x)
    
#---------------------------------------
#Pull URLs from text
def pull_urls(text):
    import re
    a_a = []
    a_a.append(re.findall(r'(https?://[^\s]+)', text))
    return a_a

#---------------------------------------
# define a function to find synonyms using WordNet

def find_synonyms(search_term):
    from nltk.corpus import wordnet as wn
    syns = wn.synsets(search_term)

    synonyms1 = []
    synonyms2 = []

    for syn in wn.synsets(search_term):
        for lem in syn.lemmas():
            synonyms1.append(lem.name())

    synonyms1 = list(set(synonyms1))
    for syn in synonyms1:
        if '_' in syn:
            syn = syn.replace('_', '-')
        synonyms2.append(syn)
    return synonyms2

#----------------------------------------
# Define a function to find synonyms using Oxforddictionaries.com

def get_synonyms(search_term):
    import  requests
    import json

    #------------- Step 1 ----------------
    # TODO: replace with your own app_id and app_key
    app_id = '0e9ea05d'
    app_key = 'ec8fab01f90f1e5469722d5c46ae1bc0'
    language = 'en'

    #------------- Step 2 ----------------
#     search_term = 'violence'#The word you are interested in searching

    #------------- Step 3 ----------------
    url = 'https://od-api.oxforddictionaries.com:443/api/v2/entries/'  + language + '/'  + search_term.lower()

    #url Normalized frequency
    urlFR = 'https://od-api.oxforddictionaries.com:443/api/v2/stats/frequency/word/'  + language + '/?corpus=nmc&lemma=' + search_term.lower()

    r = requests.get(url, headers = {'app_id' : app_id, 'app_key' : app_key})
    
    word_json_information = json.loads(r.text)#converts the pulled string into a dictionary

    #------------- Step 4 ----------------
    t = [] #establish a temp list to hold the dictionary elements

    #FOR loop to pull only the dictionary text
    for counter in range(15):
        try:
            t.append(word_json_information['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['synonyms'][counter])
        except IndexError:
            break
        except KeyError:
            break


    syns = []# establish temp list to hold the extracted synonyms

    #FOR loop to extract the synonyms from the dictionary 't' in loop above
    for counter in range(len(t)):
        g = t[counter]['text']
        syns.append(g)

    return syns

#---------------------------
# Cosine Similarity Function

def cosine_similarity(string_1, string_2):
    import string
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')

    def clean_string(text):
        text = ''.join([word for word in text if word not in string.punctuation])
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text

    corpi = [string_1, string_2]

    corpi = list(map(clean_string, corpi))

    vectorizer = CountVectorizer().fit_transform(corpi)
    vectors = vectorizer.toarray()

    cosim = cosine_similarity(vectors)

    def cosine_sim_vectors(vec1, vec2):
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    return cosine_sim_vectors(vectors[0], vectors[1])

#----------------------------------------
#Determining Levenshtein distance between two corpi

def levenshtein_distance(string_1, string_2):
    import Levenshtein
    return Levenshtein.distance(string_1, string_2)

#-------------------------------------
# Text Summarization Function written by Jeanna Schoonmaker

def text_summarization(text):
    #removing special characters
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', text )

    #tokenizing text of original article into sentences.
    sentence_list = nltk.sent_tokenize(text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
            
    #find weighted frequency of each word by dividing number of occurrences of all words by frequency of most occurring word
    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    
    #Calculate scores for each sentence by adding weighted frequencies of words in each sentence
    #50 set as maximum sentence word length

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 50: 
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
                        
    #Use heapq library to retrieve 7 sentences with highest scores

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    #Create summary by joining sentences with highest scores together in one paragraph
    summary = ' '.join(summary_sentences)
    return summary
    
#---------------------------------------
# Word Cloud

def show_wordcloud(data, title = None):
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    stopwords = set(STOPWORDS)
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=50, 
        scale=12,
        random_state=42).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# --------------------------------
# This function returns the type of encoding in a pdf
# We want to use it to improve our decoding of pdfs
# Save the encoding type into the dataframe for later recall.

def determine_pdf_encoding(file):
    import chardet 
    rawdata = open(file, "rb").read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return charenc

#------------------------------------
# TF-IDF document similarity by Eric Naon

def tfidf_string_similarity_test(correct_string,test_string):
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    two_strings = [correct_string,test_string]
    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(two_strings)
    pairwise_similarity = tfidf * tfidf.T
    arr = pairwise_similarity.toarray()
    np.fill_diagonal(arr, -1)
    return arr[0][1]

#-----------------------------------















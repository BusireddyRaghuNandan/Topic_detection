text1 = open("poem.txt", "rb").read()


# Importing Gensim
import gensim
from gensim import corpora

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string


# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
from six import iteritems
 # collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('poem.txt'))
 # remove stop words and words that appear only once
stop = set(stopwords.words('english'))
stop_ids = [dictionary.token2id[stopword] for stopword in stop
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)



class MyCorpus(object):
     def __iter__(self):
         for line in open('poem.txt'):
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(line.lower().split())


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(MyCorpus(), num_topics=3, id2word = dictionary, passes=50)


print(ldamodel.print_topics(num_topics=3, num_words=3))

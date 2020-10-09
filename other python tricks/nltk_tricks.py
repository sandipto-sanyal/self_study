# Regex pattern matching
# match the pattern word1 bla bla bla...(0 to n bla's) word2
import re
cleaned_text = 'word1 ball cat dog cow donkey word2'
re_pattern1 = 'word1\W+(?:\w+\W+){0,5}?word2'
re_pattern2 = 'word1\W+(?:\w+\W+){0,4}?word2'
print(bool(re.search(re_pattern1,cleaned_text))) # returns true
print(bool(re.search(re_pattern2,cleaned_text))) # returns true
#-------------------------------------------------------------------
# nltk lemmatization with multiple POS
import nltk
lem = nltk.stem.WordNetLemmatizer()
def multiple_lemmatization(word: str,
                           pos_tags: list = ['v', 'n']
                           ) -> str:
    '''
    Multiple POS tags for lemmatization

    Parameters
    ----------
    word : The word to lemmatize
        DESCRIPTION.
    pos_tags : list, optional
        List of POS to lemmatize. The default is ['v','n'].

    Returns
    -------
    str
        DESCRIPTION.

    '''
    for pos in pos_tags:
        word = lem.lemmatize(word, pos=pos)
    return word

text = 'Wolves and morning joggers are jogging together'
tokens = text.split()
lem_toks = [multiple_lemmatization(word.lower()) for word in tokens]
print(' '.join(lem_toks)) # Output: wolf and morning jogger be jog together

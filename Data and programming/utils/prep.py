import re


#sorting a mislabeling issue
def fixK(v):
    """taking the short form of gender classification
    and returning the long form"""
    if v == 'K':
        return 'Kvinne'
    if v == 'M':
        return 'Mann'
    else:
        return v

#sorting some mislabeling issues, converting to an approximate date 
#based on the string information about the semester.
def splitter(i):
    """Taking a semester description string on the form of 
    <YYYY season>
    and returning a list of years and seasons"""
    if i:
        test = str(i).split()
        if test[0].isnumeric() == True:
            result = [test[1],test[0]]
        else:
            result = test
        return result
    else:
        return i

def vhconverter(i):
    """taking semester season descriptions and returning an
    approximate date for the end of the semester for easier
    visualization"""
    if i == 'VÅR':
        return '-06-30'
    if i == 'HØST':
        return '-12-30'
    else:
        return 'NA'
    
def colFixer(col, repeatedText, stringCols, boolCols, addedCols,
            stringColsGuide, addedColsGuide, boolColsGuide):
    def stringReplacer(s, repeatedText):
        return s.replace(repeatedText,'').replace('  ',' ').replace(' : ',';')
    if col in stringCols:
        newCol = 'string'+stringColsGuide[col]+';'+stringReplacer(col,repeatedText)
    elif col in boolCols:
        newCol = 'bool;'+boolColsGuide[col]+';'+stringReplacer(col,repeatedText)
    elif col in addedCols:
        newCol = 'other'+addedColsGuide[col]+';'+col
    else:
        newCol = 'int;'+stringReplacer(col,repeatedText)
        
    return col+';'+newCol

def idxLabeler(c,gen):
    """taking a preprocessed document and a generator, 
    identying whether it is
    nan or a tokenized document (type list), and in case of list, 
    returns next from generator - to be used to label 
    the documents with actual
    contents and label them with its own key (as the 
    majority of the responses in this 
    case do not contain textual responses"""
    if c:
        return next(gen)
    else:
        return None
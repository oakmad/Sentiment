from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

def loadTrain():
    sw = loadStopWords()
    pos = {}
    neg = {}
    for f in movie_reviews.files():
        if "pos" in f: 
            pos[f] = loadReview(f, sw)   #load the movie review
        else:
            neg[f] = loadReview(f, sw)   #load the movie review
    r = {}
    r["+"] = pos
    r["-"] = neg                
    return r

def loadReview(f, sw):
    words = {}
    review = movie_reviews.raw(f)   #load the movie review
    for w in review.split():     #for each word in the review
        if w in sw:              #If the word is one of our stop words
            if w in words:
                words[w] = words[w] + 1
            else:
                words[w] = 1
    return words
            
def loadStopWords():
    stop = []
    for w in stopwords.words(files='english'):
        stop.append(w)
    return stop

def buildWordMap(posNeg):
    wm = {}
    i = 1
    for sym, pn in posNeg.iteritems():      #Loops over positive and negative collections
        for file, words in pn.iteritems():  #Loops over each file in the collection
            for w, count in words.iteritems(): #Loop over each word in the file
                if not w in wm:             #If the word is not already in the map
                    wm[w] = i
                    i = i + 1
    return wm 
    
def buildInvertHash(posNeg):
    negPos = {}
    i = 1
    for sym, pn in posNeg.iteritems():      #Loops over positive and negative collections
        for file, words in pn.iteritems():  #Loops over each file in the collection
            for w, count in words.iteritems(): #Loop over each word in the file
                if w in negPos:
                    negPos[w] = negPos[w] + 1
                else:
                    negPos[w] = 1
    return negPos 

def formatSVM(pn, wm, np):
    output = ""
    v = []
    tfn = 0.0
    scores = []
    for w, i  in pn.iteritems():
        v.append([wm[w], w]) #Put wordmap hash and word into array
        tfn = tfn + i
    v.sort()
    for i, w in v:
        tf = pn[w]/tfn
        df = np[w]
        if tf > 0:
            scores.append(1)
        else:
            scores.append(0)
    tscore = 0.0
    tnum = len(scores)
    if (tnum == 0.0): return ""
    for s in scores:
        tscore = tscore + s
    score_norm = tscore ** 0.5
    scout = []
    for i, w in v:
        tf = pn[w] / tfn
        df = np[w]
        try:
            if (df > 20 and scores[i] > 0):
                #scout.push([wm[pair[1]], (scores[idx] / score_norm)].join(':'))
                #scout.append([wm[w], ":".join(scores[i] / score_norm)]) 
                print str(wm[w]) + ":" + str(scores[i] / score_norm)
        except:
            pass
    #return ' '.join(scout)
    #return array.array(' ', scout).tostring() #join each item in cut with a space


def writeSVMFiles(fold, pn, wm, np):
    fTest = open("test_" + str(fold) + ".dat", 'w') 
    fTrain = open("train_" + str(fold) + ".dat", 'w')  
    for sym, pn in posNeg.iteritems():      #Loops over positive and negative collections
        if sym == "+":
            label = "1"
        else:
            label = "-1"
        i = 0
        for file, words in pn.iteritems():  #Loops over each file in the collection
            print formatSVM(words, wm, np)
            #if ((i / 100) == fold):
                # test example
            #    fte.write("#{label} #{format_item(pn[k1][fn], wm, np)}\n")
            #else:
                # training example
            #    ftr.write("#{label} #{format_item(pn[k1][fn], wm, np)}\n")
            i = i + 1

if __name__ == "__main__":
    print "Loading raw data and conducting initial parse..."
    posNeg = loadTrain() #positive and negative review data
    #print posNeg
    print "Creating Word Map"
    wordMap = buildWordMap(posNeg)
    #print wordMap
    print "Creating Invert Hash"
    negPos = buildInvertHash(posNeg)
    #print negPos
    print "Writing Folds:"
    for i in range(1, 10): #Do this 10 times
        print "               %s" % i
        writeSVMFiles(i, posNeg, wordMap, negPos)

    
    
    
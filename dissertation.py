from nltk.corpus import wordnet #used for a lot of the processing such as comparative distances between words.
from nltk.tokenize.punkt import * #used to split text into sentences
from nltk.tag.brill import * #used to determine if a word is verb, noun etc
import nltk
import urllib2 #used to connect to remote web servers and download pages
from BeautifulSoup import BeautifulSoup
from BeautifulSoup import NavigableString
import htmlentitydefs #Convert HTML entities to ISO Latin characters
import re #Regular Expression Handler
import math
from math import sqrt 
from statlib import stats
from cPickle import dump #Used to store the taggers so they don't need to be rerun
from cPickle import load
import logging #used to log actions to the disk
import os
        
class PassageAnalysis:
    "Analyzes passages of text and returns a dictionary of results" 
    corpus=""
    """positive = { 'N':wordnet.N['positive'][1], #wn.synset('positive.n.01') wn.path_similarity(wn.synset('positive.n.01'), wn.synset('true.n.01'))
                'ADJ':wordnet.ADJ['positive'][5],
                'V':wordnet.V['assert'][3],         #There is no verb for positive: assert is the best match
                'ADV':wordnet.ADV['positively'][0],
                }"""
    positive = { 'N':wordnet.synset('positive.n.01'),
                'ADJ':wordnet.synset('positive.a.05'),
                'V':wordnet.synset('assert.v.03'),         #There is no verb for positive: assert is the best match
                'ADV':wordnet.synset('positively.r.0'),
                }            
    negative = { 'N':wordnet.synset('negative.n.01'),
                'ADJ':wordnet.synset('negative.a.03'),
                'V':wordnet.synset('negative.v.0'),
                'ADV':wordnet.synset('negatively.r.0'),
                }
    scoredPassage = []
    pos_mean = 0
    neg_mean = 0
    t_score = 0
    t_prob = 0
    negations = ['not','no','never'] #these words flip the scores
    wordnet_ic = nltk.corpus.wordnet_ic.ic("ic-brown.dat")
       
    def __init__(self):
        #Initialize our dictionaries of synsets that we use to score the dictionary
        print "Initializing..."
        self.__logStatus('debug', 'Engine Starting')
        print "    -> Loading tagger Pickle"
        try:
            curr_dir = os.getcwd()
            pickle = os.path.join(curr_dir, 'ee_tagger.pkl')
            input = open(pickle, "rb")
            self.brill_tagger = load(input)
            input.close()
        except:
            try:
                input = open('/var/www/vhosts/emptyorfull.com/cgi-bin/emotion_engine/ee_tagger.pkl', "rb")
                self.brill_tagger = load(input)
                input.close()
            except:
                self.__logStatus('exception', 'Trouble loading pickle %s' % pickle)
                print "    -> Tagger doesn't exist, building..." 
                self.__buildTagger() #Build the tagger, this is EXPENSIVE
        print "Corpus Analyzer Initialized."
        
    
        
    def analyzePassage(self, text):        
        self.corpus = text
        self.__parsePassage()

    def __parsePassage(self):
        tokenize_sent = PunktSentenceTokenizer() #Sentence tokenizer
        tokenize_word = PunktWordTokenizer() #Word Tokenizer
        sentences = tokenize_sent.tokenize(self.corpus) #tokenize passage into sentences

        pos_corpus = []
        neg_corpus = []
        self.pos_n = 0
        self.neg_n = 0
        for sentence in sentences:
            #print sentence
            sentence_scores = []
            pos_tally = []
            neg_tally = []
            sent_pos_n = 0
            sent_neg_n = 0
            flip = False
            for word_tag in self.brill_tagger.tag(tokenize_word.tokenize(sentence)): #for word_tag in tokenize_word.tokenize(sentence):
                pos_score, neg_score = self.__scorePassage(word_tag[0], word_tag[1])
                if flip: #switch negative and positive scores
                    sentence_scores.append([neg_score, pos_score])
                else:
                    sentence_scores.append([pos_score, neg_score])
                    
                if word_tag[0] in self.negations: #from now on flip scores
                    if flip:
                        flip = False
                    else:
                        flip = True
                        
            for score in sentence_scores:
                if score[0] != None: 
                    pos_tally.append(score[0])
                    pos_corpus.append(score[0])
                if score[1] != None: 
                    neg_tally.append(score[1])
                    neg_corpus.append(score[1])
                if score[0] > score[1]:
                    sent_pos_n = sent_pos_n + 1
                    self.pos_n =  self.pos_n + 1
                elif score[0] < score[1]:
                    sent_neg_n = sent_neg_n + 1
                    self.neg_n = self.neg_n + 1
            try:
                #TTest_Ind calculates our scores and probability of make an error in 5% of cases
                sen_t_score, sen_t_prob = stats.ttest_ind(pos_tally, neg_tally) 
            except: #A zero division error
                sen_t_score, sen_t_prob = 0, 0
            try:
                sent_pos_mean = stats.mean(pos_tally)
            except:
                sent_pos_mean = 0
            try:
                sent_neg_mean = stats.mean(neg_tally)
            except:
                sent_neg_mean = 0
            self.scoredPassage.append({'sentence':sentence, 
                                       'pos_mean':sent_pos_mean, 
                                       'neg_mean':sent_neg_mean,
                                       'pos_n':sent_pos_n,
                                       'neg_n':sent_neg_n, 
                                       't_score':sen_t_score , 
                                       't_prob':sen_t_prob}) #append the sentence and its scores
        #Calculate the T-Score
        self.pos_mean = stats.mean(pos_corpus)
        self.neg_mean = stats.mean(neg_corpus)
        try:
            self.t_score, self.t_prob = stats.ttest_ind(pos_corpus, neg_corpus)
        except: #A zero division error
            self.t_score, self.t_prob = 0, 0
        
        print "Finished Parsing and scoring"
        
                
    def __scorePassage(self, _word, _part):
        """
        Every word was analyzed in WordNet to discover its statistical relationship to our root words
        We use these values to baseline scores so there is no built in bias towards one extreme or the other
        POSITIVE::::::
            Nouns: 
              -> 76692 scores found
              -> mean: 0.0632666699072
              -> st dev: 0.0119508088962 
            Verbs: 
              -> 3146 scores found
              -> mean: 0.144374642452
              -> st dev: 0.0561323108116 
        NEGATIVE::::::
            Nouns: 
              -> 76692 scores found
              -> mean: 0.0783480664081
              -> st dev: 0.0231938765085 
            Verbs: 
              -> 1378 scores found
              -> mean: 0.121428670104
              -> st dev: 0.059387288084 
        
        """
        pos_score = None
        neg_score = None
       
        #Generate our score looking by comparing words for parts of speech, nouns are most common, they become the catch all, not best performance wise
        # See this page for possible part values based on brown corpus http://alias-i.com/lingpipe/docs/api/com/aliasi/corpus/parsers/BrownPosParser.html
        if _part in ["AP","AP$","DT","DT$","DTI","DTS","DTX","JJ","JJ$","JJR","JJS","JJT","WQL"]: #Adjectives
            for word_syn in wordnet.synsets(_word, pos=wordnet.ADJ): #Lets assume we want the most extreme word
                #print word_syn.lemma_names
                x = self.__similarity(word_syn, self.positive['ADJ'])
                if x > pos_score: pos_score = x
                x = self.__similarity(word_syn, self.negative['ADJ'])
                if x > neg_score: neg_score = x
        elif _part in ["VB","VBD","VBG","VBN","VBZ"]: #Verbs
            for word_syn in wordnet.synsets(_word, pos=wordnet.VERB): #Lets assume we want the most extreme word
                #print word_syn.lemma_names
                x = self.__similarity(word_syn, self.positive['V'])
                if x > pos_score: pos_score = x
                x = self.__similarity(word_syn, self.negative['V'])
                if x > neg_score: neg_score = x
            #Normalize the scores by calculating the z score
            if pos_score != None: pos_score = (pos_score - 0.144374642452) / 0.0561323108116
            if neg_score != None: neg_score = (neg_score - 0.121428670104) / 0.059387288084
        elif _part in ["MD","NR","NR$","NRS","RB","RB$","RBR","RBT","RN","RP", "WRB"]: #Adverb
            for word_syn in wordnet.synsets(_word, pos=wordnet.ADV): #Lets assume we want the most extreme word
                #print word_syn.lemma_names
                x = self.__similarity(word_syn, self.positive['ADV'])
                if x > pos_score: pos_score = x
                x = self.__similarity(word_syn, self.negative['ADV'])
                if x > neg_score: neg_score = x
        else: #Probably a noun
            for word_syn in wordnet.synsets(_word, pos=wordnet.NOUN): #Lets assume we want the most extreme word
                #print word_syn.lemma_names
                x = self.__similarity(word_syn, self.positive['N'])
                if x > pos_score: pos_score = x
                x = self.__similarity(word_syn, self.negative['N'])
                if x > neg_score: neg_score = x
            #Normalize the scores by calculating the z score
            #if pos_score != None: pos_score = (pos_score - 0.0632666699072) / 0.0119508088962
            #if neg_score != None: neg_score = (neg_score - 0.0783480664081) / 0.0231938765085
        
        if pos_score != None: pos_score = math.fabs(pos_score) 
        if neg_score != None: neg_score = math.fabs(neg_score) 
        #print "word: %s part: %s negative: %s positive %s" % (_word, _part, neg_score, pos_score)
        return [pos_score, neg_score] #send back the scores
     
    def __similarity(self, word, compareto):
        try:
            score = word.jcn_similarity(compareto, self.wordnet_ic, True)
        except:
            score = wordnet.path_similarity(word, compareto)
        if score == -1: score = None #No path between the words was found
        return score
    
    def __buildTagger(self):
        self.__logStatus('debig', 'Building Pickle')
        #Build our text taggers, these define words via their parts of speech: Noun, Adjective, Verb, Adverb 
        brown_train = nltk.corpus.brown.tagged_sents() #used to train the taggers on words and their meanings
        #Hierachy of taggers, from least to most effective, fallback if words not found is the regex tagger
        regexp_tagger = nltk.RegexpTagger(
                                          [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
                                           (r'(The|the|A|a|An|an)$', 'AT'),   # articles
                                           (r'.*able$', 'JJ'),                # adjectives
                                           (r'.*ness$', 'NN'),                # nouns formed from adjectives
                                           (r'.*ly$', 'RB'),                  # adverbs
                                           (r'.*s$', 'NNS'),                  # plural nouns
                                           (r'.*ing$', 'VBG'),                # gerunds
                                           (r'.*ed$', 'VBD'),                 # past tense verbs
                                           (r'.*', 'NN')                      # nouns (default)
                                           ])
        unigram_tagger_2 = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
        #Set the rules that the Brill tagger will follow
        templates = [
                     SymmetricProximateTokensTemplate(ProximateTagsRule, (1,1)),
                     SymmetricProximateTokensTemplate(ProximateTagsRule, (2,2)),
                     SymmetricProximateTokensTemplate(ProximateTagsRule, (1,2)),
                     SymmetricProximateTokensTemplate(ProximateTagsRule, (1,3)),
                     SymmetricProximateTokensTemplate(ProximateWordsRule, (1,1)),
                     SymmetricProximateTokensTemplate(ProximateWordsRule, (2,2)),
                     SymmetricProximateTokensTemplate(ProximateWordsRule, (1,2)),
                     SymmetricProximateTokensTemplate(ProximateWordsRule, (1,3)),
                     ProximateTokensTemplate(ProximateTagsRule, (-1, -1), (1,1)),
                     ProximateTokensTemplate(ProximateWordsRule, (-1, -1), (1,1)),
                     ]
        trainer = FastBrillTaggerTrainer(initial_tagger=unigram_tagger_2,templates=templates, trace=3,deterministic=True)
        self.brill_tagger = trainer.train(brown_train, max_rules=10)
        #Pickle the tagger to a file
        #try:
        curr_dir = os.getcwd()
        pickle = os.path.join(curr_dir, 'ee_tagger.pkl')
        output = open(pickle, "w")
        dump(self.brill_tagger, output, -1)
        output.close()
        #except: #Keep going tagger will be rebuilt each time
        #    print "Couldn't save PICKLE!!!!!"

    def __logStatus(self, type, message):
        curr_dir = os.getcwd()
        logfile = os.path.join(curr_dir, 'emotion_engine.log')
        logging.basicConfig(filename=logfile, level=logging.DEBUG, 
                        format="%(asctime)s [%(levelname)s] %(message)s")
        if type == "debug":
            logging.debug(message)
        else:
            logging.exception(message)
        
             
class UrlAnalysis(PassageAnalysis):
    "Retrieves a URL and analyzes the text. Inherits PassageAnalysis"
    
    def analyzeUrl(self, _url):    
        self.__url = _url
        self.__corpus = ""
        request = urllib2.Request(self.__url, None,{'User-Agent':'Mozilla/5.0(Windows; U; Windows NT 5.1; en-US; rv:1.6) Gecko/20040113'})
        self.__soup = BeautifulSoup(urllib2.urlopen(request).read())
        self.__parseUrl(self.__soup.body)
        self.analyzePassage(self.__corpus)
        
    def __parseUrl(self, tags):
        if not isinstance(tags, NavigableString): #to avoid recursive issues
            for tag in tags:
                if tag.__class__ == NavigableString and tag.parent.name != "script":
                    self.__corpus += tag.__str__()
                else:
                    self.__parseUrl(tag)
         
    def __convertHtml(self, passage):
        return re.sub(r'&(#?)(.+?);',self.__convertEntity,passage) 
    
    def __convertEntity(self, matches):
        if matches.group(1)=='#':
            try:
                if int(matches.group(2)) <= 255:  #Its ASCII
                    return chr(int(matches.group(2)))
                else:                       #Its unicode
                    return unichr(int(matches.group(2)))
            except ValueError:
                return '&#%s;' % matches.group(2) #send it back its unknown
        else:
            try:
                return htmlentitydefs.entitydefs[matches.group(2)] #match it to one of the known entities
            except KeyError:
                #return '&%s;' % m.group(2)
                pass
        
    def __loadPassageFromDB(self, id):
        self.passage_id = id
        #Get the sentences to evaluate
        self.db_cursor.execute ("""SELECT url
                            FROM analysis_passages
                            WHERE id = %s""", (self.passage_id))
        row = self.db_cursor.fetchone ()
        self.__url = row["url"]
        
if __name__ == "__main__":
    print "Main, you shouldn't be here....."
    
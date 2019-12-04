import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd


# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

from email import policy
from email.policy import default
from email.parser import BytesParser
from bs4 import BeautifulSoup

# THIS IN PYTHON (if not done yet)
import nltk

nltk.download('stopwords')

# THIS IN COMMAND LINE (if not done yet)
# python -m spacy download en  # run in terminal once

# NLTK Stop words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(
    ['www', 'http', '.com', 'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know',
     'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy',
     'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may',
     'take', 'come'])
stop_words.extend(
    ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst",
     "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", 
     "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost",
      "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce",
       "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", 
       "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth",
        "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", 
        "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", 
        "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C",
         "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci",
          "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering",
           "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date",
            "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does",
             "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3",
              "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", 
              "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", 
              "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", 
              "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", 
              "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full",
               "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj",
                "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", 
                "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby",
                 "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", 
                 "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", 
                 "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information",
                  "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy",
                   "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", 
                   "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf",
                    "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M",
                     "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", 
                     "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", 
                     "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither",
                      "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", 
                      "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", 
                      "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", 
                      "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", 
                      "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", 
                      "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", 
                      "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", 
                      "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", 
                      "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively",
                       "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv",
                        "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", 
                        "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns",
                         "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", 
                         "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", 
                         "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", 
                         "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx",
                          "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore",
                           "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", 
                           "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus",
                            "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries",
                             "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", 
                             "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully",
                              "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", 
                              "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", 
                              "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", 
                              "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", 
                              "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X",
                               "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd",
                                "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])


# https://stackoverflow.com/questions/30565404/remove-all-style-scripts-and-html-tags-from-an-html-page/30565420
def cleanMe(html):
    soup = BeautifulSoup(html)  # create a new bs4 object from the html data loaded
    for script in soup(["script", "style"]):  # remove all javascript and stylesheet code
        script.extract()
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        yield (sent)

    # !python3 -m spacy download en  # run in terminal once


def process_words(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out


def format_topics_sentences(ldamodel, corpus, texts, rawEmail, subjects, fromMail, emailFileNames):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    contentsRaw = pd.Series(rawEmail)
    contentsSubjects = pd.Series(subjects)
    contentsFrom = pd.Series(fromMail)
    contentsFileName = pd.Series(emailFileNames)
    sent_topics_df = pd.concat([contentsFileName, sent_topics_df, contents, contentsSubjects, contentsFrom, contentsRaw], axis=1)
    return (sent_topics_df)


def EmailToDf(listPathEmail):
    data = []
    subjects = []
    fromMail = []
    successFullEmails = []
    success = 0
    failure = 0
    for pathEmail in listPathEmail:
        with open(pathEmail, "rb") as myfile:
            print(pathEmail)
            try:
                msg = BytesParser(policy=policy.default).parse(myfile)

                contentType = msg['Content-Type']  # html or plain text

                body = msg.get_body(preferencelist=(contentType)).get_content()

                subjects.append(msg['subject'])
                fromMail.append(msg['from'])
                if "html" in contentType.lower():  # extract content form html
                    body = cleanMe(body)
                    print(f"Part of the email of length {len(body)}: {(body)[:50]}")

                data.append(body) 
                successFullEmails.append(pathEmail) # store all email paths that are valid
                success += 1
            except Exception as e:
                failure += 1
                print("Error in " + str(pathEmail) + str(e))  # some email could not be parsed
    print("-----------------------------------")
    print(str(success) + " emails successfully parsed. ")
    print(str(failure) + " email failed. ")
    df = pd.DataFrame(data)
    
    return df, data, subjects, fromMail, successFullEmails


def PredictEmails(listPathEmail, trainedLDAModel):
    df, rawEmail, subjects, fromMail, successFullEmails = EmailToDf(listPathEmail)
    if len(df.index > 0):
        # Convert to list
        data = df[0].values.tolist()
        data_words = list(sent_to_words(data))

        data_ready = process_words(data_words, stop_words)  # processed Text Data!
        #print(data_ready)

        # Create Dictionary
        id2word = corpora.Dictionary(data_ready)

        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_ready]

        lda_model = trainedLDAModel #gensim.models.ldamodel.LdaModel.load(pathLDAModel)

        df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, data_ready, rawEmail, subjects, fromMail, successFullEmails)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'FileName', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Subjects',
                                    'From', 'Raw Email']
        return (df_dominant_topic)
    else:
        return pd.DataFrame() # something went wrong. return empty dataframe


def TrainModel(listPathEmail):
    df, _, _, _, _ = EmailToDf(listPathEmail)
    if len(df.index > 0):
        # Convert to list
        data = df[0].values.tolist()
        data_words = list(sent_to_words(data))

        data_ready = process_words(data_words, stop_words)  # processed Text Data!
        #print(data_ready)

        # Create Dictionary
        id2word = corpora.Dictionary(data_ready)

        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_ready]

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=3, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=10,
                                                per_word_topics=True)
        
        print(lda_model.print_topics())

        return (lda_model)
    else:
        return None # something went wrong. return empty

def ManualTopicModellingPostProcess(dfProcessedEmail):
    countChangeToTopic3 = 0
    countChangeToTopic2 = 0
    countChangeToTopic1 = 0
    for i in dfProcessedEmail.index:
        if dfProcessedEmail.at[i, 'Dominant_Topic'] == 0:
            mainWords = dfProcessedEmail.at[i, 'Text']

            wordsToCheckSelling = ["viagra", "pill", "health", "cialis", "ciali", "shop", "pharmacy", "medicine",
                                   "antidepressants", "vitamins", "impotence"]
            wordsToCheckSex = ["sex", "fuck"]
            wordsToCheckBlackMail = ["hack", "compensation", "payment", "infect"]

            if any(elem in wordsToCheckSelling for elem in mainWords):
                dfProcessedEmail.at[i, 'Dominant_Topic'] = 3
                dfProcessedEmail.at[i, 'Keywords'] = "Manually Changed Topic"
                countChangeToTopic3 += 1
            elif any(elem in wordsToCheckSex for elem in mainWords):
                dfProcessedEmail.at[i, 'Dominant_Topic'] = 2
                dfProcessedEmail.at[i, 'Keywords'] = "Manually Changed Topic"
                countChangeToTopic2 += 1
            elif any(elem in wordsToCheckBlackMail for elem in mainWords):
                dfProcessedEmail.at[i, 'Dominant_Topic'] = 1
                dfProcessedEmail.at[i, 'Keywords'] = "Manually Changed Topic"
                countChangeToTopic1 += 1
    print("In Post Process Changed " + str(countChangeToTopic3) + " to topic 3.")
    print("In Post Process Changed " + str(countChangeToTopic2) + " to topic 2.")
    print("In Post Process Changed " + str(countChangeToTopic1) + " to topic 1.")
    return dfProcessedEmail

# NOT USED
# This is not working properlty right now, but its not needed in the current version anyways
def CreateTopicMapping(lda_model):
    t =lda_model.print_topics()
    print(t)
    d = {}
    for topicIdx in range(3):
        # check for first topic (we assume that this one is easy to detect)
        check = True
        for w in ["girl", "sex"]:
            check = check and w in t[topicIdx][1]
        if check:
            d["girl"] = topicIdx
            continue

        # check second topic (this one is not so easy and shall have many variations)
        check = True
        for w in ["account", "payment"]:
            check = check and w in t[topicIdx][1]
        if check:
            d["blackmail"] = topicIdx
            continue

        for w in ["bitcoin", "payment"]:
            check = check and w in t[topicIdx][1]
        if check:
            d["blackmail"] = topicIdx
            continue

        for w in ["screenshot", "payment"]:
            check = check and w in t[topicIdx][1]
        if check:
            d["blackmail"] = topicIdx
            continue
        
        # no topic match found. So this one is a undecided topic
        d["undecided"] = topicIdx
    d["selling"] = 3 # this is a manually added topic that wil be used later in the "ManualTopicModellingPostProcess" function
    return d



def ParseEmails(files, pathEmailDatasets):
    # Topics
    # [(0,
    #  '0.024*"video" + 0.014*"screen" + 0.013*"letter" + 0.012*"contact" + '
    #  '0.009*"read" + 0.009*"case" + 0.009*"time" + 0.009*"program" + '
    #  '0.009*"watch" + 0.008*"send"'),
    # (1,
    #  '0.027*"device" + 0.024*"site" + 0.023*"bitcoin" + 0.022*"screenshot" + '
    #  '0.021*"contact" + 0.019*"address" + 0.018*"payment" + 0.017*"account" + '
    #  '0.015*"amount" + 0.014*"understand"'),
    # (2,
    #  '0.111*"girl" + 0.077*"sex" + 0.055*"site" + 0.043*"fuck" + 0.042*"find" + '
    #  '0.041*"look" + 0.041*"hot" + 0.037*"ready" + 0.037*"city" + '
    #  '0.036*"tonight"')]

    # Topic 1 = Blackmail
    # Topic 2 = Sex
    # Topic 0 = undecided -> manual processing needed
    # Topic 3 = selling (after manual processing)
    
    print("Training Model")
    pathEmailDatasets = glob.glob(pathEmailDatasets+"/*.eml")
    ldaModel = TrainModel(pathEmailDatasets)

    print("Using Model for Pediction")
    dfParsed = PredictEmails(files, ldaModel)

    # print(dfParsed)
    dfParsed = ManualTopicModellingPostProcess(dfParsed)
    # print(dfParsed)
    # dfParsed.to_csv('allEmails_3Topics_PostProcess.csv', sep='\t', encoding='utf-8')
    return dfParsed

#import glob
#paths = glob.glob("spam/*.eml")
#paths = glob.glob("D:/OneDriveYeritsyan/OneDrive/Studium/Maastricht/IS/Project/spam/*.eml")
#res = ParseEmails(paths, "spam")
#print(res[['FileName', 'Dominant_Topic', 'Raw Email']])
#res.to_csv("testNewDataset.csv", sep='\t')
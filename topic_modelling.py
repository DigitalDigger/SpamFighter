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
                    print(body)

                data.append(body)
            except Exception as e:
                print("Error in " + str(pathEmail) + str(e))  # some email could not be parsed

    df = pd.DataFrame(data)
    return df, data, subjects, fromMail


def PredictEmails(listPathEmail, pathLDAModel):
    df, rawEmail, subjects, fromMail = EmailToDf(listPathEmail)
    if len(df.index > 0):
        # Convert to list
        data = df[0].values.tolist()
        data_words = list(sent_to_words(data))

        data_ready = process_words(data_words, stop_words)  # processed Text Data!
        print(data_ready)

        # Create Dictionary
        id2word = corpora.Dictionary(data_ready)

        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_ready]

        lda_model = gensim.models.ldamodel.LdaModel.load(pathLDAModel)

        df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, data_ready, rawEmail, subjects, fromMail, listPathEmail)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'FileName', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Subjects',
                                    'From', 'Raw Email']
        return (df_dominant_topic)
    else:
        return pd.DataFrame() # something went wrong. return empty dataframe


def ManualRopicModellingPostProcess(dfProcessedEmail):
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


def ParseEmails(files):
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

    dfParsed = PredictEmails(files, "trainedLDAModel_3Topics.mod")
    # print(dfParsed)
    dfParsed = ManualRopicModellingPostProcess(dfParsed)
    # print(dfParsed)
    # dfParsed.to_csv('allEmails_3Topics_PostProcess.csv', sep='\t', encoding='utf-8')
    return dfParsed

#import glob
#res = ParseEmails(glob.glob("spam/*.eml"))
#print(res)
#res.to_csv("test.csv", sep='\t')
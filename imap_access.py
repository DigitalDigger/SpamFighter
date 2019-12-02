#!/usr/bin/env python
#
import sys
import imaplib
import email
from email import policy
from email.parser import BytesParser
import glob
import codecs
import struct
import re
import time
from flask import Flask, request, jsonify
import logging
import threading
import smtplib, ssl
from email.mime.text import MIMEText
from email.utils import parseaddr


import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
from pprint import pprint
import os

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
stop_words.extend(['www', 'http', '.com', 'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

# https://stackoverflow.com/questions/30565404/remove-all-style-scripts-and-html-tags-from-an-html-page/30565420
def cleanMe(html):
    soup = BeautifulSoup(html) # create a new bs4 object from the html data loaded
    for script in soup(["script", "style"]): # remove all javascript and stylesheet code
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
        yield(sent)  

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
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

def format_topics_sentences(ldamodel, corpus, texts, rawEmail, subjects, fromMail):
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
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    contentsRaw = pd.Series(rawEmail)
    contentsSubjects = pd.Series(subjects)
    contentsFrom = pd.Series(fromMail)
    sent_topics_df = pd.concat([sent_topics_df, contents, contentsSubjects, contentsFrom, contentsRaw], axis=1)
    return(sent_topics_df)

def EmailToDf(listPathEmail):
    data = []
    subjects = []
    fromMail = []
    for pathEmail in listPathEmail:
        with open (pathEmail, "rb") as myfile:
            print(pathEmail)
            try:
                msg = BytesParser(policy=policy.default).parse(myfile)
                
                contentType= msg['Content-Type'] # html or plain text
                
                body = msg.get_body(preferencelist=(contentType)).get_content()
                
                subjects.append(msg['subject'])
                fromMail.append(msg['from'])
                if "html" in contentType.lower(): # extract content form html
                    body = cleanMe(body)
                    print(body)

                data.append(body)
            except Exception as e:
                print("Error in " + str(pathEmail) + str(e)) # some email could not be parsed

    df = pd.DataFrame(data)
    return df, data, subjects, fromMail

def PredictEmails(listPathEmail, pathLDAModel):
    df, rawEmail, subjects, fromMail = EmailToDf(listPathEmail)
    
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

    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, data_ready, rawEmail, subjects, fromMail)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Subjects', 'From', 'Raw Email']
    return(df_dominant_topic)

def ManualRopicModellingPostProcess(dfProcessedEmail):
    countChangeToTopic3 = 0
    countChangeToTopic2 = 0
    countChangeToTopic1 = 0
    for i in dfProcessedEmail.index:
        if dfProcessedEmail.at[i,'Dominant_Topic'] == 0:
            mainWords = dfProcessedEmail.at[i,'Text']
            
            wordsToCheckSelling = ["viagra", "pill", "health", "cialis","ciali", "shop", "pharmacy", "medicine", "antidepressants", "vitamins", "impotence"]
            wordsToCheckSex = ["sex", "fuck"]
            wordsToCheckBlackMail = ["hack", "compensation", "payment", "infect"]

            if any(elem in wordsToCheckSelling for elem in mainWords):
                dfProcessedEmail.at[i,'Dominant_Topic'] = 3
                dfProcessedEmail.at[i,'Keywords'] = "Manually Changed Topic"
                countChangeToTopic3 += 1
            elif any(elem in wordsToCheckSex for elem in mainWords):
                dfProcessedEmail.at[i,'Dominant_Topic'] = 2
                dfProcessedEmail.at[i,'Keywords'] = "Manually Changed Topic"
                countChangeToTopic2 += 1
            elif any(elem in wordsToCheckBlackMail for elem in mainWords):
                dfProcessedEmail.at[i,'Dominant_Topic'] = 1
                dfProcessedEmail.at[i,'Keywords'] = "Manually Changed Topic"
                countChangeToTopic1 += 1
    print("In Post Process Changed " + str(countChangeToTopic3) + " to topic 3.")
    print("In Post Process Changed " + str(countChangeToTopic2) + " to topic 2.")
    print("In Post Process Changed " + str(countChangeToTopic1) + " to topic 1.")
    return dfProcessedEmail
    

def ParseEmails(files):
    # Topics
    #[(0,
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

    #Topic 1 = Blackmail
    #Topic 2 = Sex
    #Topic 0 = undecided -> manual processing needed

    dfParsed = PredictEmails(files, "trainedLDAModel_3Topics.mod")
    #print(dfParsed)
    dfParsed = ManualRopicModellingPostProcess(dfParsed)
    #print(dfParsed)
    #dfParsed.to_csv('allEmails_3Topics_PostProcess.csv', sep='\t', encoding='utf-8')
    return dfParsed


'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

import requests
import random
import string  # to process standard python strings
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('popular', quiet=True)  # for downloading packages

a = "a1"
b = ["sex", "girl", "woman", "want", "money"]
c = "Hey handsome. I have just, only  watched your pictures. You are very attrractive. Iam so, very bored tonight and I want 2 offer u talking. My profile is http://dating-future69.com/?b9af34-1&sid=3D6 here."
d = "get with me f#ckbuddy"


def mail_body_generator(a, b):
    #Topic 1 = Blackmail
    #Topic 2 = Sex
    #Topic 0 = undecided -> manual processing needed
    # Set up clustering options and get cluster value
    a = int(a[0])
    dict1 = {2: "Sex proposal", 1: "Extorting money", 0: "Product offer", 3: "Malicious Software"}
    dict_engager = {2: "More pictures", 1: "Issue", 0: "Product", 3: "Software"}
    cluster = dict1[a]
    words_bag_sentence = " ".join(b) + "."
    subject = d
    greeting_0 = "Hello, sadly I dont speak in english GOOD. {} , I say yes. Provide more information. ".format(
        dict_engager[a])

    # This list need to be updated with random greetings
    greeting_list = []
    input_mail = greeting_0 + words_bag_sentence + "Please provide more information about {}. Yours faithfully, John".format(
        dict_engager[a])
    return input_mail


def mail_genearator(a, b, c, d):
    input_mail = mail_body_generator(a, b)
    a = int(a[0])

    # This element in next verswion will be ranodmly chosen
    dict_engager = {2: "More pictures", 1: "Issue", 0: "Product", 3: "Software"}
    greeting_0 = "Hello, sadly I dont speak in english GOOD. {} , I say yes. Provide more information. ".format(
        dict_engager[a])
    greeting_1 = greeting_0
    engager = "Please provide more information about {}. Yours faithfully, John".format(dict_engager[a])

    # this lement takes the main function and as inout takes global variable state _dict
    text_body = text_generator(state_dict, input_mail)
    sent_tokens = nltk.sent_tokenize(text_body)  # converts to list of sentences
    word_tokens = nltk.word_tokenize(text_body)  # converts to list of words
    clean_tokens = []
    for i in sent_tokens:
        if i.find("<") != -1 or i.find(">") != -1 or i.find("Anonymous") != -1:
            pass
        else:
            clean_tokens.append(i)
    print(sent_tokens, file=open("results\gpt_2_output.txt", "a+"))

    if os.path.exists('results'):
        pass
    else:
        os.mkdir("results")

    print(greeting_1, file=open("results\mail_text.txt", "a"))
    print(clean_tokens[1], file=open("results\mail_text.txt", "a"))
    print(clean_tokens[2], file=open("results\mail_text.txt", "a"))
    print(engager, file=open("results\mail_text.txt", "a"))

    print(greeting_1 + clean_tokens[1] + clean_tokens[2] + engager)

    return greeting_1 + '\n' + clean_tokens[1] + '\n' + clean_tokens[2] + '\n' + engager


def text_generator(state_dict, input_mail):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=False)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    args = parser.parse_args()
    text = input_mail
    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(text)
    context_tokens = enc.encode(text)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens if not args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                pass
                # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            return (text)


"""
        Hello, 
        sadly I dont speak in english GOOD. More pictures , YES! Provide more INFO <3 :D
        I just want what I need to play and to have fun.
        Please note that you can change your profile picture in the menu below.
        Please provide more information about More pictures. 
        Yours faithfully, 
        John
"""

IMAP_SERVER = 'imap.spaceweb.ru'
SMTP_SERVER = 'smtp.spaceweb.ru'
EMAIL_ACCOUNT = "info@infoprocsoft.com"
EMAIL_FOLDER = "Spam"
OUTPUT_DIRECTORY = './spam'

PASSWORD = 'skipidar'

logger = logging.getLogger('SpamFighter')
logger.setLevel(logging.DEBUG)


print(struct.calcsize("P") * 8)

def sanitize_encoding(enc):
    try:
        if enc is None:
            return
        codecs.lookup(enc)
        return enc
    except LookupError:
        try:
            enc = enc.replace('-','')
            codecs.lookup(enc)
            return enc
        except LookupError:
            # Not a thing, either way
            return None

pattern_uid = re.compile('\d+ \(UID (?P<uid>\d+)\)')

def parse_uid(data):
    match = pattern_uid.match(data.decode("utf-8") )
    return match.group('uid')

def respond_email(receiver, message):
    context = ssl.create_default_context()
    port = 465
    with smtplib.SMTP_SSL(SMTP_SERVER, port, context=context) as server:
        server.login(EMAIL_ACCOUNT, PASSWORD)
        server.sendmail(EMAIL_ACCOUNT, receiver, message)
        server.quit()

def move_email(M, email_id, folder):
    resp, data = M.fetch(email_id, "(UID)")
    msg_uid = parse_uid(data[0])

    result = M.uid('COPY', msg_uid, folder)

    if result[0] == 'OK':
        mov, data = M.uid('STORE', msg_uid, '+FLAGS', '(\Deleted)')
        M.expunge()

def process_mailbox(M):
    """
    Dump all emails in the folder to files in output directory.
    """

    rv, data = M.search(None, "ALL")
    if rv != 'OK':
        print("No messages found!")
        return

    for num in data[0].split():
        rv, data = M.fetch(num, '(RFC822)')

        if rv != 'OK':
            print("ERROR getting message", num)
            return
        print("Writing message ", num)
        f = open('%s/%s.eml' %(OUTPUT_DIRECTORY, num), 'wb')
        f.write(data[0][1])
        f.close()
        with open('%s/%s.eml' %(OUTPUT_DIRECTORY, num), 'rb') as fp:  # select a specific email file from the list
            msg = BytesParser(policy=policy.default).parse(fp)
            msg["_charset"] = sanitize_encoding(msg["_charset"])
            try:
                text = msg.get_body(preferencelist=('plain, html')).get_content()
            except:
                move_email(M, num, 'SpamFailed')
                continue
            print(text)  # print the email content
            return_path = EMAIL_ACCOUNT
            from_email = EMAIL_ACCOUNT
            sender_email = EMAIL_ACCOUNT
            for item in msg._headers:
                if item[0].find('Return-path') != -1:
                    return_path = item[1]
                elif item[0].find('From') != -1:
                    from_email = item[1]
                elif item[0].find('Sender') != -1:
                    sender_email = item[1]

            if parseaddr(return_path)[1] != '':
                receiver = parseaddr(return_path)[1]
            elif parseaddr(from_email)[1] != '':
                receiver = parseaddr(from_email)[1]
            elif parseaddr(sender_email)[1] != '':
                receiver = parseaddr(sender_email)[1]
            receiver = EMAIL_ACCOUNT

            result = ParseEmails(['%s/%s.eml' %(OUTPUT_DIRECTORY, num)])

            generated = mail_genearator(result['Dominant_Topic'].tolist(), result['Keywords'].tolist(), result['Raw Email'].tolist(), result['Subjects'].tolist())

            respond_email(receiver, generated)
            with open("output.txt", "a", encoding="utf-8") as myfile:
                myfile.write(text)



            move_email(M, num, 'SpamProcessed')
            time.sleep(5)


def main():
    M = imaplib.IMAP4_SSL(IMAP_SERVER)
    M.login(EMAIL_ACCOUNT, PASSWORD)
    rv, data = M.select(EMAIL_FOLDER)
    if rv == 'OK':
        print("Processing mailbox: ", EMAIL_FOLDER)
        process_mailbox(M)
        M.close()
        print("Done processing mailbox: ", EMAIL_FOLDER)
    else:
        print("ERROR: Unable to open mailbox ", rv)
    M.logout()


app = Flask(__name__)
@app.route('/api/v1/getEmails', methods=['GET'])
def get_emails():
    # API to be implemented
    if request.is_json:
        req = request.get_json()
        print(req)


def set_interval(func, sec):
    def func_wrapper():
        func()
        set_interval(func, sec)

    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t



if __name__ == "__main__":

    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()

    timeoutInSeconds = 0
    # second command line argument is timeout in seconds for checking
    # for emails
    if len(sys.argv) > 2:
        timeoutInSeconds = int(sys.argv[2])
    if timeoutInSeconds == 0:
        main()
    else:
        set_interval(main, timeoutInSeconds)
        logger.info('Starting server...')
        app.run(host='0.0.0.0', port=10120)


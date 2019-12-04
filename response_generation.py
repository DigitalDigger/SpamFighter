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
import globals

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
    dict_engager = {2: "nice pictures", 1: "the issue", 0: "product", 3: "software"}
    GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me", "Dear friend,"]
    greeting_0 = "Hello, sadly I dont speak in english GOOD. {} , I say yes. Provide more information. ".format(dict_engager[a])
    greeting_1 = random.choice(GREETING_RESPONSES) + '\n' + "Thanks for your mail about {} and reaching out to me. ".format(dict_engager[a])
    engager_0 = "Please provide more information about {}. ".format(dict_engager[a])
    ending = "Yours faithfully, John"
    Engager_RESPONSES = ["As always thanks for keeping in touch. How I can help?","The wise man once said that we say BYE for by your eye. What your eye is looking for?","Hope to hear from you soon with more info.","Nevertheless thanks for your time and maybe update me soon","Hope to hear from you soon with more info.","Please I need to know more ASAP",]
    engager_1 = random.choice(Engager_RESPONSES)
    
    #this lement takes the main function and as inout takes global variable state _dict
    text_body = text_generator(state_dict, input_mail )
    sent_tokens = nltk.sent_tokenize(text_body)# converts to list of sentences 
  
    
    clean_tokens = []
    for i in sent_tokens:
        if i.find("<") != -1 or i.find(">") != -1 or i.find("Anonymous") != -1:
            pass
        else:
            clean_tokens.append(i)
            
    final_respond = ""    
    if e == True:
        final_respond =  '\n' + greeting_0 + '\n' + clean_tokens[1] + '\n' + clean_tokens[2] + '\n' + engager_0 + ending
        
    elif e == False:
        final_respond = '\n' + greeting_1 + '\n' + clean_tokens[1] + '\n' + clean_tokens[2] + '\n' + engager_1 + ending
    else:
        final_respond = "5th input variable is not in Boolean."
    return final_respond 
    
    """
    VERSION FOR STORING outputs as file
    print(sent_tokens, file=open("results\gpt_2_output.txt", "a"))
    
    if os.path.exists('results'):
        pass
    else:
        os.mkdir("results")
    
    print(greeting_1, file=open("results\mail_text.txt", "a"))
    print(clean_tokens[1], file=open("results\mail_text.txt", "a"))        
    print(clean_tokens[2], file=open("results\mail_text.txt", "a")) 
    print(engager, file=open("results\mail_text.txt", "a"))
    """


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

    #print(text)
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
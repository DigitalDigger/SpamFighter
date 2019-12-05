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
from pprint import pprint
import os
import torch
from envelopes import Envelope, GMailSMTP

from topic_modelling import ParseEmails
from response_generation import mail_genearator
import globals

IMAP_SERVER = 'imap.spaceweb.ru'
SMTP_SERVER = 'smtp.spaceweb.ru'
EMAIL_ACCOUNT = "info@infoprocsoft.com"
EMAIL_FOLDER = "Spam"
OUTPUT_DIRECTORY = './spam'

PASSWORD = 'WeFightSpammers!123'

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

def send_email(receiver, subject, body, source):
    envelope = Envelope(
        from_addr=(EMAIL_ACCOUNT),
        to_addr=(receiver),
        subject=subject,
        text_body=body
    )
    envelope.add_attachment(source)

    # Send the envelope using an ad-hoc connection...
    envelope.send(SMTP_SERVER, login=EMAIL_ACCOUNT,
                  password=PASSWORD, tls=True)

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
            # receiver = EMAIL_ACCOUNT

            result = ParseEmails(['%s/%s.eml' %(OUTPUT_DIRECTORY, num)], '%s/' %(OUTPUT_DIRECTORY))

            if len(result.index) > 0:
                subject = result['Subjects'].tolist()

                if len(subject) > 0:
                    subject = subject[0]
                else:
                    subject = "RE: "

                generated = mail_genearator(result['Dominant_Topic'].tolist(), result['Text'].tolist()[0], result['Raw Email'].tolist(), result['Subjects'].tolist(), True)

                send_email(receiver, subject, generated, '%s/%s.eml' %(OUTPUT_DIRECTORY, num))
                with open("output.txt", "a", encoding="utf-8") as myfile:
                    myfile.write(text)
                    myfile.write(generated)



                move_email(M, num, 'SpamProcessed')
            time.sleep(50)


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
        globals.state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
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


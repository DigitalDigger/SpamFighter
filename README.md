## **SpamFighter powered by topic modelling and GPT2**

**Spam the spammers**


Spam emails represent a significant share of all emails circulating in the world. They are used for a wide range of purposes: from product advertising up to phishing activities when malcreants try to steal your confidential information.

To tackle this issue, most of the email service providers make use of dedicated filters to block spam. Effectiveness of such filters may vary, e.g., Google claimed in 2015 that only 0.1% of emails that end up in your gmail inbox folder is spam. However, despite the high success rates of the modern spam filters the spam industry still functions and induces growing costs to businesses. Statistics demonstrate that even a single response to 12.5 million spam emails still allows spammers to earn around $7000 per day.

Our approach is different from the passive defense that is exploited by spam filters. In this work we applied an active defense, i.e., we automatically generate responses to the spam emails and sent them back to the spammers. Such responses are usually processed manually by malcreants which induces additional workload and a waste of time of the spammers. The idea is similar to the denial-of-service attack but it is applied to the human resources in our case.

## Quick Start

To run the code, please deploy pre-trained GPT-2-pytorch model first by following the instructions available here:

[GPT-2-PyTorch deployment](https://github.com/graykode/gpt-2-Pytorch/blob/master/README.md)
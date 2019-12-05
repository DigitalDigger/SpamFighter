## **SpamFighter powered by topic modelling and GPT2**

**Spam the spammers**


Spam  emails  represent  a  significant  share  of  all  emailscirculating  in  the  world.  They  are  used  for  a  wide  range  ofpurposes:  from  product  advertising  up  to  phishing  activitieswhen malcreants try to steal your confidential information.To  tackle  this  issue,  most  of  the  email  service  providersmake  use  of  dedicated  filters  to  block  spam.  Effectivenessof  such  filters  may  vary,  e.g.,  Google  claimed  in  2015  thatonly  0.1%  of  emails  that  end  up  in  your  gmail  inbox  folderis  spam. 

However,  despite  the  high  success  rates  of  themodern  spam  filters  the  spam  industry  still  functions  andinduces  growing  costs  to  businesses.  Statistics  demonstratethat  even  a  single  response  to  12.5  million  spam  emails  stillallows spammers to earn around $7000 per day.

Our  approach  is  different  from  the  passive  defense  that  isexploited  by  spam  filters.  In  this  work  we  applied  an  activedefense, i.e., we automatically generate responses to the spamemails  and  sent  them  back  to  the  spammers.  Such  responsesare usually processed manually by malcreants which inducesadditional workload and a waste of time of the spammers. Theidea is similar to the denial-of-service attack but it is appliedto the human resources in our case.In  order  to  interest  spammers  in  responding  our  automati-cally generated emails, we apply the most recent advances inthe domain of the natural language processing, namely, deeplanguage generative models.

## Quick Start

To run the code, please deploy pre-trained GPT-2-pytorch model first by following the instructions available here:

[GPT-2-PyTorch deployment](https://github.com/graykode/gpt-2-Pytorch/blob/master/README.md)
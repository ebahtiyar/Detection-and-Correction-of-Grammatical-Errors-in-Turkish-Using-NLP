# Detection and Correction of Grammatical Errors in Turkish Using NLP techniques

In this project, grammatical errors in Turkish are detected and corrected using NLP techniques and traditional algorithms. Trained models and algorithms have been turned into a website to be user-friendly. The website is also built using the Django Framework.

The grammatical errors that can be detected are:

Errors due to lack of punctuation marks
Errors due to suffix or conjunction in /da/
Errors caused by the writing of case suffixes
Character based errors

In addition, it can suggest Turkish origin words instead of foreign origin words.

4 different models and a large root dictionary are the basis of this project.
Approximately 3 million sentences trained in these models were collected from news sites to consist of up-to-date words. These sentences were parsed according to certain requirements (such as rooting words, separation from numbers) and the models were trained.

The dictionary is composed of 3 million sentences collected.

While collecting the sentences, packages such as Selenium and BeautifulSoup were used. Stored using SQLite3
 



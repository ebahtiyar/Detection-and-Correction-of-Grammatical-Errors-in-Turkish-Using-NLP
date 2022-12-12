# Detection and Correction of Grammatical Errors in Turkish Using NLP techniques

In this project, grammatical errors in Turkish are detected and corrected using NLP techniques and traditional algorithms. Trained models and algorithms have been turned into a website to be user-friendly. The website is also built using the Django Framework.

The grammatical errors that can be detected are:

*Errors due to lack of punctuation marks

*Errors due to suffix or conjunction in da/de/ki

*Errors caused by the writing of case suffixes

*Character based errors

*In addition, it can suggest Turkish origin words instead of foreign origin words.

4 different models and a large root dictionary are the basis of this project.
Approximately 3 million sentences trained in these models were collected from news sites to consist of up-to-date words. These sentences were parsed according to certain requirements (such as rooting words, separation from numbers) and the models were trained.

The dictionary is composed of 3 million sentences collected.

While collecting the sentences, packages such as Selenium and BeautifulSoup were used. Stored using SQLite3

For more detailed information, you can access the project report from the link below.
https://docs.google.com/document/u/0/d/1oi831nig5W-wDC6lKFIczQmA6bqnYWCU/mobilebasic

Images from the project are here

![ss_1](https://user-images.githubusercontent.com/46243758/207122225-9b713abc-1948-4ddf-8c07-8fdd970f60a7.png)

![ss_2](https://user-images.githubusercontent.com/46243758/207122236-ac600aaf-1733-4709-9198-e430dc729f93.png)

![ss_3](https://user-images.githubusercontent.com/46243758/207122249-758585a0-9cdb-4067-87ac-d397a0c12b1d.png)

Project Prensentation Video:
https://www.youtube.com/watch?v=UOAqi54JK_8&ab_channel=EmreBahtiyar

Project Models and Dictionary Driver Link:
https://drive.google.com/drive/folders/16rH72NwdoqrRBjzT8e3o4L6uPrBA9pqw?usp=share_link

Use of:
1) Download all the model and side files from the google driver link above
2) Edit the model and dictionary paths in the code according to your own computer
3) Run the website by typing python run manage.py in the terminal. That is all :)

You can reach me about the project from anywhere.
 



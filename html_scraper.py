import newspaper
import json
import requests
import csv
from datetime import datetime

def dateify(date):
    date_string = date[0:10]
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    new_date_string = date_object.strftime("%m-%d-%Y")
    return new_date_string

# some code from https://newscatcherapi.com/blog/python-web-scraping-libraries-to-mine-news-data
def html_scraper(url):
    if(len(url.split(" ")) != 1):
        return url

    article = newspaper.Article(url=url, language='en')
    print(type(article))
    article.download()
    print(type(article))
    article.parse()
    print(type(article))
    article ={
    "title": str(article.title),
    "text": str(article.text),
    "authors": article.authors,
    "published_date": str(article.publish_date),
    "top_image": str(article.top_image),
    "videos": article.movies,
    "keywords": article.keywords,
    "summary": str(article.summary)
    }

    return article['text']

def add_to_CSV(url, classification):
    if(len(url.split(" ")) != 1):
        return

    article = newspaper.Article(url=url, language='en')
    article.download()
    article.parse()
    article ={
    "title": str(article.title),
    "text": str(article.text),
    "authors": article.authors,
    "published_date": str(article.publish_date),
    "top_image": str(article.top_image),
    "videos": article.movies,
    "keywords": article.keywords,
    "summary": str(article.summary)
    }
    new_row = [article['title'], article['text'], classification, dateify[article['published_date']]]

    with open('new_data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

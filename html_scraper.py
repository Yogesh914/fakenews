import newspaper
import json

# some code from https://newscatcherapi.com/blog/python-web-scraping-libraries-to-mine-news-data
def html_scraper(url):
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

    return article['title'], article['text']

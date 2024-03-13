import requests
from bs4 import BeautifulSoup

url = 'https://www.bis.org/publ/bisbull35.htm'
response = requests.get(url)

html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

# Find the first <meta content element
citation_author_elements = soup.find_all('meta', {'name': 'citation_author'})
citation_authors = [meta['content'] for meta in citation_author_elements]

citation_publication_date_elements = soup.find_all('meta', {'name': ['citation_publication_date']})
citation_publication_date = [meta['content'] for meta in citation_publication_date_elements]

print()


from langchain.document_loaders import WebBaseLoader

import os
os.environ['APIFY_API_TOKEN'] = '**'

web_links = ["https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1168",
    "https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1167",
    "https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1164"]
loader = WebBaseLoader(web_links)
documents = loader.load()


#return documents
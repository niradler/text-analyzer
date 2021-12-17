import nltk
import yake
import os
import re
from pydantic import BaseModel
from starlette.responses import RedirectResponse, JSONResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader, APIKey
from fastapi import Security, Depends, FastAPI, HTTPException, Header
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

nltk.download('words')
en_words = set(nltk.corpus.words.words())

API_KEY = os.getenv('API_KEY', 'test')
API_KEY_NAME = os.getenv('KEY_NAME', 'access_token')
COOKIE_DOMAIN = os.getenv('DOMAIN', 'localhost')

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

language = "en"
max_ngram_size = 2
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 15

kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                     dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)


async def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
    api_key_cookie: str = Security(api_key_cookie),
):

    if api_key_query == API_KEY:
        return api_key_query
    elif api_key_header == API_KEY:
        return api_key_header
    elif api_key_cookie == API_KEY:
        return api_key_cookie
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )


@app.get("/")
async def home():
    return "Is up."


@app.get("/openapi.json", tags=["documentation"])
async def get_open_api_endpoint(api_key: APIKey = Depends(get_api_key)):
    response = JSONResponse(
        get_openapi(title="FastAPI security test",
                    version=1, routes=app.routes)
    )
    return response


@app.get("/api", tags=["documentation"])
async def get_documentation(api_key: APIKey = Depends(get_api_key)):
    response = get_swagger_ui_html(
        openapi_url=f"/openapi.json?{API_KEY_NAME}={api_key}", title="docs")
    response.set_cookie(
        API_KEY_NAME,
        value=api_key,
        domain=COOKIE_DOMAIN,
        httponly=True,
        max_age=1800,
        expires=1800,
    )
    return response


@app.get("/logout")
async def logout_and_remove_cookie():
    response = RedirectResponse(url="/")
    response.delete_cookie(API_KEY_NAME, domain=COOKIE_DOMAIN)
    return response


class Options(BaseModel):
    text: str
    words: Optional[list] = None


def removeNoneEnglish(text, words):
    if words is None:
        words = []
    tagsAndWords = set(words + list(en_words))
    return " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in tagsAndWords or not w.isalpha())


def cleanhtml(raw_html, words):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext.replace("&nbsp;", " ")
    cleantext = removeNoneEnglish(cleantext, words)
    return cleantext


@app.post("/text/analyze")
async def analyze_text(options: Options, api_key: APIKey = Depends(get_api_key)):
    text = cleanhtml(options.text, options.words)
    keywords = kw_extractor.extract_keywords(text)
    keywords = [keyword[0] for keyword in keywords]

    return {'keywords': keywords}

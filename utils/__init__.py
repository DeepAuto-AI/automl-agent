import requests

from bs4 import BeautifulSoup
from urllib.parse import unquote
from serpapi import GoogleSearch
from kaggle.api.kaggle_api_extended import KaggleApi

from openai import OpenAI
from configs import AVAILABLE_LLMs


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_kaggle():
    api = KaggleApi()
    api.authenticate()
    return api


# def search_web(query):
#     try:
#         # Abort the request after 10 seconds
#         response = requests.get(f"https://www.google.com/search?hl=en&q={query}")
#         response.raise_for_status()  # Raises an HTTPError for bad responses
#         html_string = response.text
#     except requests.exceptions.RequestException as e:
#         print_message(
#             "system",
#             "Request Google Search Failed with " + str(e) + "\n Using SerpAPI.",
#         )
#         params = {
#             "engine": "google",
#             "q": query,
#             "api_key": "",
#         }

#         search = GoogleSearch(params)
#         results = search.get_dict()
#         return results["organic_results"]

#     # Parse the HTML content
#     soup = BeautifulSoup(html_string, "html.parser")

#     # Find all <a> tags
#     links = soup.find_all("a")

#     if not links:
#         raise Exception('Webpage does not have any "a" element')

#     # Filter and process the links
#     filtered_links = []
#     for link in links:
#         href = link.get("href")
#         if href and href.startswith("/url?q=") and "google.com" not in href:
#             cleaned_link = unquote(
#                 href.split("&sa=")[0][7:]
#             )  # Remove "/url?q=" and split at "&sa="
#             filtered_links.append(cleaned_link)

#     # Remove duplicates and prepare the output
#     unique_links = list(set(filtered_links))
#     return {"organic_results": [{"link": link} for link in unique_links]}[
#         "organic_results"
#     ]

def search_web(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "your api key",
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results["organic_results"]


def print_message(sender, msg, pid=None):
    pid = f"-{pid}" if pid else ""
    sender_color = {
        "user": color.PURPLE,
        "system": color.RED,
        "manager": color.GREEN,
        "model": color.BLUE,
        "data": color.DARKCYAN,
        "prompt": color.CYAN,
        "operation": color.YELLOW,
    }
    sender_label = {
        "user": "💬 You:",
        "system": "⚠️ SYSTEM NOTICE ⚠️\n",
        "manager": "🕴🏻 Agent Manager:",
        "model": f"🦙 Model Agent{pid}:",
        "data": f"🦙 Data Agent{pid}:",
        "prompt": "🦙 Prompt Agent:",
        "operation": f"🦙 Operation Agent{pid}:",
    }

    msg = f"{color.BOLD}{sender_color[sender]}{sender_label[sender]}{color.END}{color.END} {msg}"
    print(msg)
    print()


def get_client(llm: str = "qwen"):
    if llm.startswith("gpt"):
        return OpenAI(api_key=AVAILABLE_LLMs[llm]["api_key"])
    else:
        return OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )

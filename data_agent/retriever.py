import requests
import re

from openai import OpenAI
from utils import get_kaggle, search_web, print_message
from utils.embeddings import chunk_and_retrieve
from configs import AVAILABLE_LLMs
from validators import url
from glob import glob


def retrieve_datasets(user_requirements, data_path, client, model):
    """Retrieve up-to-date and state-of-the-art knowledge from the websearch and relevant hubs for data manipulation and analysis."""

    # Return detailed sources and implementation for subsequent code generation
    # Internal: User Upload, User Link, and Data Hub
    # External: OpenML, UCI ML Dataset Archive, Hugging Face, Torch DataHub, Tensorflow DataHub, and Kaggle Dataset
    datasets = []

    for data in user_requirements["dataset"]:
        data["task"] = user_requirements["problem"]["downstream_task"]
        data["llm_client"] = client
        data["llm_model"] = model
        if data.get("source", "user-upload") in ["user-upload", "upload"]:
            # get upload files from storage
            loader_key = True
            datasets.append(
                {
                    "name": data["name"],
                    "loader_key": data_path,
                    "source": "user-upload",
                }
            )
            continue
        elif data.get("source", "user-upload") == "user-link":
            # download file from the given link
            data["url"] = re.search(
                "(?P<url>https?://[^\s]+)", str(user_requirements["dataset"])
            ).group("url")
            loader_key = retrieve_download(**data)
            datasets.append(
                {
                    "name": data["name"],
                    "loader_key": loader_key,
                    "source": "user-link",
                }
            )
            continue
        elif data.get("source", "user-upload") == "direct-search":
            # search using name in data loaders
            loader_key = retrieve_huggingface(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "huggingface-hub",
                    }
                )
                continue

            loader_key = retrieve_kaggle(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "kaggle-hub",
                    }
                )
                continue

            loader_key, hub_name = retrieve_pytorch(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": hub_name,
                    }
                )
                continue

            loader_key = retrieve_tensorflow(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "tensorflow-datasets",
                    }
                )
                continue

            loader_key = retrieve_uci(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "source": "ucimlrepo",
                    }
                )
                continue

            loader_key = retrieve_openml(**data)
            if loader_key:
                datasets.append({"name": data["name"], "source": "openml"})
                continue
              
        else:
            loader_key = retrieve_infer(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "infer-search",
                    }
                )
                continue
    return datasets


def _is_applicable(data_task, user_task):
    if isinstance(data_task, list):
        for i in range(len(data_task)):
            data_task[i] = (
                data_task[i].replace("-", " ").replace("_", " ").lower().strip()
            )
            user_task = (
                user_task.replace("-", " ").replace("_", " ").lower().strip()
                if isinstance(user_task, str)
                else [
                    task.replace("-", " ").replace("_", " ").lower().strip()
                    for task in user_task
                ]
            )
            if data_task[i] in user_task:
                return True
    elif isinstance(data_task, str):
        if data_task in user_task:
            return True
    return False


def retrieve_infer(**kwargs):
    from langchain_community.document_loaders import PDFMinerLoader, AsyncChromiumLoader
    from langchain_community.document_transformers import Html2TextTransformer

    query_prompt = f"""Give me a search query without special symbols to search for a dataset described by "{kwargs['description']}". Give me only the search query without explanation."""
    client = kwargs["llm_client"]

    messages = [
        {
            "role": "system",
            "content": "You are a data curator who has a lot of experience in data collection.",
        },
        {"role": "user", "content": query_prompt},
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model=kwargs["llm_model"], messages=messages, temperature=0.3
            )
            break
        except Exception as e:
            print_message("system", e)
            continue

    search_query = response.choices[0].message.content.strip().replace('"', "")
    kaggle_api = get_kaggle()
    datasets = kaggle_api.datasets_list(search=search_query, sort_by="votes")[:10]
    for dataset in datasets:
        tags = [tag["name"] for tag in dataset["tags"]]
        if _is_applicable(tags, kwargs["modality"]):
            return dataset
    else:
        search_results = search_web(search_query)
        DOMAIN_BLOCKLIST = [
            "youtube.com",
            "twitter.com",
            "x.com",
            "hindawi.com",
            "ejournal.ittelkom-pwt.ac.id",
        ]

        search_results = [
            result
            for result in search_results
            if not any(domain in result["link"] for domain in DOMAIN_BLOCKLIST)
        ][:10]
        urls = [link["link"] for link in search_results if url(link["link"])]
        loader = AsyncChromiumLoader([link for link in urls if ".pdf" not in link])
        html = loader.load()

        html2text = Html2TextTransformer()
        html_docs = html2text.transform_documents(html)
        for link in urls:
            if (
                "arxiv.org/pdf" in link
                or "/pdf?id=" in link
                or "&name=pdf" in link
            ):
                try:
                    html_docs += PDFMinerLoader(link).load()
                except Exception as e:
                    print('cannot load', link, 'with error:', e)                   
                
        # generate context from HTML pages for summary
        context = "".join(
            [
                d.page_content
                for d in chunk_and_retrieve(
                    ref_text=kwargs["description"],
                    documents=html_docs,
                    top_k=10,
                    ranker="bm25",
                )
            ]
        )
        # genearte summmary
        summary_prompt = f"""I searched the web using the query: {search_query}. Here is the result:
        =====================
        {context}
        =====================
        
        According to the given result, where can I retrieve the dataset described by the following reference data description? Please give me one location or possible URL for download.
        # Reference Data Description
        {kwargs['description']}
        """

        while True:
            try:
                response = client.chat.completions.create(
                    model=kwargs["llm_model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data curator who has a lot of experience in data collection.",
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.3,
                )
                break
            except Exception as e:
                print_message("system", e)
                continue
        
        if re.search("(?P<url>https?://[^\s]+)", response.choices[0].message.content.strip()):
            return re.search("(?P<url>https?://[^\s]+)", response.choices[0].message.content.strip()).group("url")
        else:
            return response.choices[0].message.content.strip()

def retrieve_huggingface(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        from huggingface_hub import HfApi

        hf_api = HfApi()
        dataset = list(
            hf_api.list_datasets(
                search=kwargs["name"],
                sort="downloads",
                full=True,
            )
        )
        if len(dataset) > 0:
            dataset = dataset[0]
            if dataset.card_data:
                if _is_applicable(dataset.card_data.task_categories, kwargs["task"]) or _is_applicable(dataset.card_data.task_ids, kwargs["task"]):
                    return dataset
                else:
                    return None
            else:
                return None
        else:
            return None


def retrieve_tensorflow(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        import tensorflow_datasets as tfds

        name = kwargs["name"].strip().lower()
        is_exist = name in [
            ds.replace("_", " ").replace("-", " ").strip()
            for ds in tfds.list_builders()
        ]
        if is_exist:
            dataset = tfds.list_builders().index(name)
            return dataset
    return None


def retrieve_pytorch(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        utility_classes = [
            "DatasetFolder",
            "ImageFolder",
            "VisionDataset",
            "wrap_dataset_for_transforms_v2",
        ]
        if "image" in kwargs["modality"] or "video" in kwargs["modality"]:
            from torchvision import datasets

            avail_datasets = [
                fn.lower().replace("-", " ").replace("_", " ")
                for fn in datasets.__all__
                if fn not in utility_classes
            ]
            hub_name = "torchvision"
        elif "text" in kwargs["modality"]:
            from torchtext import datasets

            avail_datasets = [
                fn.lower().replace("-", " ").replace("_", " ")
                for fn in datasets.__all__
                if fn not in utility_classes
            ]
            hub_name = "torchtext"
        elif "audio" in kwargs["modality"]:
            from torchaudio import datasets

            avail_datasets = [
                fn.lower().replace("-", " ").replace("_", " ")
                for fn in datasets.__all__
                if fn not in utility_classes
            ]
            hub_name = "torchaudio"
        elif "graph" in kwargs["modality"]:
            from torch_geometric import datasets

            avail_datasets = [
                fn.lower().replace("-", " ").replace("_", " ")
                for fn in datasets.__all__
                if fn not in utility_classes
            ]
            hub_name = "torch_geometric"
        else:
            # not support multimodal, time series, and tabular modalities
            return None, None

    query = kwargs["name"].lower().replace("-", " ").replace("_", " ")
    if query in avail_datasets:
        return avail_datasets[avail_datasets.index(query)], hub_name
    else:
        return None, None


def retrieve_kaggle(**kwargs):
    kaggle_api = get_kaggle()
    if kwargs["name"] and kwargs["name"] != "":
        datasets = kaggle_api.datasets_list(search=kwargs["name"], sort_by="votes")[:10]
        for dataset in datasets:
            tags = [tag["name"] for tag in dataset["tags"]]
            if _is_applicable(tags, kwargs["modality"]) or _is_applicable(
                tags, kwargs["task"]
            ):
                return kaggle_api.metadata_get(*dataset["ref"].split("/"))
        else:
            return None


def retrieve_uci(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        from ucimlrepo import fetch_ucirepo

        try:
            dataset = fetch_ucirepo(name=kwargs["name"])
            if dataset != None:
                return dataset
        except:
            return None


def retrieve_openml(**kwargs):
    from openml.datasets import list_datasets

    if kwargs["name"] and kwargs["name"] != "":
        datalist = list_datasets(output_format="dataframe")
        found_dataset = datalist[datalist["name"].str.contains(kwargs["name"].lower())]
        found_dataset = found_dataset[found_dataset["status"] == "active"]
        return len(found_dataset) > 0


def retrieve_download(**kwargs):
    res = requests.get(kwargs["url"])
    if res.status_code == 200:
        return kwargs["url"]
    else:
        return None

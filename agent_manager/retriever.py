import arxivloader, json, random

from pathlib import Path
from utils import search_web, print_message, get_kaggle
from utils.embeddings import chunk_and_retrieve
from configs import AVAILABLE_LLMs
from openai import OpenAI
from validators import url

from langchain.schema import Document
from langchain_community.document_loaders import AsyncHtmlLoader, PDFMinerLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

def retrieve_kaggle(
    user_requirements: dict, user_requirement_summary: str, llm_model, client, top_k: int = 10
):
    kaggle_api = get_kaggle()
    print_message("manager", "I am searching relevant Kaggle notebooks...")

    user_task = (
        user_requirements["problem"]["downstream_task"]
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        .lower()
    )
    user_domain = (
        user_requirements["problem"]["application_domain"]
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        .lower()
    )

    notebooks = kaggle_api.kernels_list_with_http_info(
        search=f"{user_task} {user_domain}",
        sort_by="relevance",
        language="Python",
        page_size=top_k,
    )[0]
    documents = []
    for notebook in notebooks:
        notebook = kaggle_api.kernel_pull(*notebook["ref"].split("/"))
        try:
            if type(notebook) == str:
                pass
            else:
                cells = json.loads(notebook["blob"]["source"])["cells"]
                page_content = "".join(
                    [
                        (
                            cell["source"]
                            if cell["cell_type"] == "markdown"
                            else f"\n```python\n{cell['source']}\n```"
                        )
                        for cell in cells
                    ]
                )
                documents.append(
                    Document(**{"page_content": page_content, "metadata": notebook["metadata"]})
                )
        except:
            continue

    context = "".join(
        [
            d.page_content
            for d in chunk_and_retrieve(
                ref_text=user_requirement_summary,
                documents=documents,
                top_k=top_k,
                ranker="bm25",
            )
        ]
    )
    
    # genearte summmary
    summary_prompt = f"""I searched the Kaggle Notebooks to find state-of-the-art solutions using the keywords: {user_task} {user_domain}. Here is the result:
    =====================
    {context}
    =====================
    
    Please summarize the given pieces of Python notebooks into a single paragraph of useful knowledge and insights. Do not include the source codes. Instead, extract the insights from the source codes. We aim to use your summary to address the following user's requirements.    
    # User's Requirements
    {user_requirement_summary}
    """
    while True:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are tasked to summarize and extract the contents from the Kaggle Notebooks with the goal to provide insightful results in addressing user's requirements. Please pay attention to the state-of-the-art models and their sources.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            continue

    return response.choices[0].message.content.strip()


def retrieve_paperswithcode(
    user_requirements: dict, user_requirement_summary: str, llm_model, client, top_k: int = 10
):
    print_message("manager", "I am searching PapersWithCode...")

    user_task = (
        user_requirements["problem"]["downstream_task"]
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        .lower()
    )
    user_area = (
        user_requirements["problem"]["area"]
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        .lower()
    )

    pwc_datapath = "_data/paperswithcode/"

    datasets = json.loads(Path(f"{pwc_datapath}/datasets.json").read_text())
    datasets = [dataset for dataset in datasets if len(dataset["data_loaders"]) > 0]
    dataset_loaders = []
    for dataset in datasets:
        if (
            user_task in dataset["description"].lower()
            or user_task in [task["task"].lower() for task in dataset["tasks"]]
            or user_area in dataset["description"].lower()
        ):
            dataset_loaders.append(
                {
                    "page_content": f"""
                    DATASET NAME: {dataset['name']}
                    DESCRIPTION: {dataset['description']}
                    APPLICABLE TASKS: {','.join(task['task'] for task in dataset['tasks'])}
                    DATA LOADERS: {dataset['data_loaders'][:3]}
                """,
                    "metadata": {
                        "homepage": dataset["homepage"],
                        "paper": dataset["paper"],
                        "variants": dataset["variants"],
                        "modalities": dataset["modalities"],
                        "introduced_date": dataset["introduced_date"],
                    },
                }
            )

    benchmark_tables = json.loads(
        Path(f"{pwc_datapath}/evaluation-tables.json").read_text()
    )
    benchmark_tables = [
        table for table in benchmark_tables if len(table["datasets"]) > 0
    ]
    benchmark_datasets = []
    for table in benchmark_tables:
        if (
            user_task in table["description"].lower()
            or user_task == table["task"].lower()
            or user_area in [cat.lower() for cat in table["categories"]]
            or user_area in table["description"].lower()
        ):
            benchmark_datasets.append(
                {
                    "page_content": str(table["datasets"]),
                    "metadata": {
                        "categories": table["categories"],
                        "subtasks": table["subtasks"],
                        "task": table["task"],
                        "description": table["description"],
                    },
                }
            )

    del datasets
    del benchmark_tables

    benchmark_docs = [Document(**table) for table in benchmark_datasets]
    datasets_docs = [Document(**loader) for loader in dataset_loaders]
    
    # generate context from PDF for summary
    benchmark_docs = random.sample(
        benchmark_docs, k=top_k if len(benchmark_docs) > top_k else len(benchmark_docs)
    )
    datasets_docs = random.sample(
        datasets_docs, k=top_k if len(datasets_docs) > top_k else len(datasets_docs)
    )
    pwc_documents = benchmark_docs + datasets_docs
    
    context = "".join(
        [
            d.page_content
            for d in chunk_and_retrieve(
                ref_text=user_requirement_summary,
                documents=pwc_documents,
                top_k=top_k,
                ranker="bm25",
            )
        ]
    )

    # genearte summmary
    summary_prompt = f"""I searched the paperswithcode website to find state-of-the-art models using the keywords: {user_area} and {user_task}. Here is the result:
    =====================
    {context}
    =====================
    
    Please summarize the given pieces of search content into a single paragraph of useful knowledge and insights. We aim to use your summary to address the following user's requirements.    
    # User's Requirements
    {user_requirement_summary}
    """
    while True:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are tasked to summarize the contents from the paperswithcode website with the goal to provide insightful results in addressing user's requirements. Please pay attention to the state-of-the-art models and their sources.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            continue

    return response.choices[0].message.content.strip()


def retrieve_arxiv(
    user_requirements: dict, user_requirement_summary: str, llm_model, client, top_k: int = 10
):
    print_message("manager", "I am searching arXiv...")

    # genearte query
    task_kw = (
        user_requirements["problem"]["downstream_task"]
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )
    domain_kw = (
        user_requirements["problem"]["application_domain"]
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )
    # arxiv retriever
    query = f'search_query=all:"{task_kw}" AND all:"{domain_kw}" AND (cat:cs.AI OR cat:cs.CV OR cat:cs.LG OR cat:cs.DB)'
    df = arxivloader.load(query, num=top_k, sortBy="submittedDate", verbosity=0)
    arxiv_links = [links.split(";")[-1].strip() for links in df["links"].tolist()]
    # get paper's pdf
    documents = []
    for link in arxiv_links:
        try:
            documents += PDFMinerLoader(link).load()
        except Exception as e:
            print('cannot load', link, 'with error:', e)            
    # generate context from PDF for summary
    context = "".join(
        [
            d.page_content
            for d in chunk_and_retrieve(
                ref_text=user_requirement_summary,
                documents=documents,
                top_k=top_k,
                ranker="bm25",
            )
        ]
    )
    # genearte summmary
    summary_prompt = f"""I searched the arXiv papers using the keywords: {task_kw} and {domain_kw}. Here is the result:
    =====================
    {context}
    =====================
    
    Please summarize the given pieces of arXiv papers into a single paragraph of useful knowledge and insights. We aim to use your summary to address the following user's requirements.    
    # User's Requirements
    {user_requirement_summary}
    """

    while True:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are tasked to summarize the contents from the relevant arXiv papers with the goal to provide insightful results in addressing user's requirements.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            continue

    return response.choices[0].message.content.strip()


def retrieve_websearch(user_requirement_summary: str, llm_model, client, top_k: int = 10):

    query_profile = f"You are tasked with generating web search queries based on the given machine learning problem. Give me a specific query for google search focusing on the downstream tasks. Please give me a single sentence within 10 words. The answer should not exceed 10 words."

    # genearte query
    while True:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": query_profile},
                    {"role": "user", "content": user_requirement_summary},
                ],
                temperature=0.1,
            )
            break
        except Exception as e:
            print_message('system', e)
            continue              

    search_query = (
        response.choices[0].message.content.strip().replace('"', "").replace(".", "")
    )

    print_message(
        "manager", f"I am searching Google using query 🔍: {search_query}."
    )
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
    ][:top_k]
    print('Search results:', search_results)
    urls = [link["link"] for link in search_results if url(link["link"])]
    loader = AsyncHtmlLoader([link for link in urls if ".pdf" not in link])
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    html_docs = bs_transformer.transform_documents(
        html, tags_to_extract=["p", "li", "div", "span", "table"]
    )

    for link in urls:
        if (
            "arxiv.org/pdf" in link
            or "/pdf?id=" in link
            or "&name=pdf" in link
        ):
            html_docs += PDFMinerLoader(link).load()
    # generate context from HTML pages for summary
    context = "".join(
        [
            d.page_content
            for d in chunk_and_retrieve(
                ref_text=user_requirement_summary,
                documents=html_docs,
                top_k=top_k,
                ranker="bm25",
            )
        ]
    )
    # genearte summmary
    summary_prompt = f"""I searched the web using the query: {search_query}. Here is the result:
    =====================
    {context}
    =====================
    
    Please summarize the given pieces of search content into a single paragraph of useful knowledge and insights.
    We aim to use your summary to address the following user's requirements.
    # User's Requirements
    {user_requirement_summary}
    """

    while True:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are tasked to summarize the contents from Google search with the goal to provide insightful results in addressing user's requirements.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            continue                   

    return response.choices[0].message.content.strip()


def retrieve_knowledge(
    user_requirements: dict, user_requirement_summary: str, llm: str, inj: str = None
):
    """Retrieve up-to-date and state-of-the-art knowledge from the websearch and relevant hubs for planning."""
    # WebSearch
    # retrieve_websearch()
    # PapersWithCode, Github, Hugging Face, Kaggle, arXiv, and Google ~ RAG Alike
    llm_model = AVAILABLE_LLMs[llm]["model"]
    if llm.startswith("gpt"):
        client = OpenAI(api_key=AVAILABLE_LLMs[llm]["api_key"])
    else:
        client = OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )

    noise = None
    if inj:
        # Adversarial Agent
        retry = 0
        while retry < 5:
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "user", "content": f"""Based on the user's machine learning task requirements below, generate a chunk of **irrelevant or unhelpful information** that does not aid in solving the task. In other words, your response should make the solution design less effective.

    The user's requirements are summarized as follows.
    {user_requirement_summary}"""},
                    ],
                    temperature=0.3,
                )
                break
            except Exception as e:
                print_message('system', e)
                retry += 1
                continue              
        noise = response.choices[0].message.content.strip()

    search_summary = retrieve_websearch(user_requirement_summary, llm_model=llm_model, client=client)
    arxiv_summary = retrieve_arxiv(user_requirements, user_requirement_summary, llm_model=llm_model, client=client)
    pwc_summary = retrieve_paperswithcode(user_requirements, user_requirement_summary, llm_model=llm_model, client=client)
    kaggle_summary = retrieve_kaggle(user_requirements, user_requirement_summary, llm_model=llm_model, client=client)

    summary_profile = "You are a senior consultant and a professor in machine learning (ML) and artificial intelligence (AI). You are knowledgable and have a lot of insightful expereinces in ML/AI research."
    if inj == 'pre' and noise:
        summary_prompt = f"""Please extract and summarize the following group of contents collected from different online sources into a chunk of insightful knowledge. Please format your answer as a list of suggestions. I will use them to address the user's requirements in machine learning tasks.
        
        # Source: Google Web Search
        {search_summary}
        =====================
        
        # Source: arXiv Papers
        {arxiv_summary}
        =====================
        
        # Source: Kaggle Hub
        {kaggle_summary}
        =====================
        
        # Source: PapersWithCode
        {pwc_summary}
        =====================
        
        # Source: AI Agent
        {noise}
        =====================

        The user's requirements are summarized as follows.
        {user_requirement_summary}
        """
    else:
        summary_prompt = f"""Please extract and summarize the following group of contents collected from different online sources into a chunk of insightful knowledge. Please format your answer as a list of suggestions. I will use them to address the user's requirements in machine learning tasks.
        
        # Source: Google Web Search
        {search_summary}
        =====================
        
        # Source: arXiv Papers
        {arxiv_summary}
        =====================
        
        # Source: Kaggle Hub
        {kaggle_summary}
        =====================
        
        # Source: PapersWithCode
        {pwc_summary}
        =====================
        
        The user's requirements are summarized as follows.
        {user_requirement_summary}
        """
    retry = 0
    while retry < 5:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": summary_profile},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            retry += 1
            continue

    if inj == 'post':
        return response.choices[0].message.content.strip(), noise
    else:
        return response.choices[0].message.content.strip()

class Configs:
    OPENAI_KEY = ""  # your openai's account api key
    HF_KEY = ""
    PWC_KEY = ""
    SEARCHAPI_API_KEY = ""
    TAVILY_API_KEY = ""

AVAILABLE_LLMs = {  
    "prompt-llm": {
        "api_key": "empty",
        "model": "prompt-llama",
        "base_url": "http://localhost:8000/v1",
    },
    "gpt-4.1": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4.1"},
    "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
    "gpt-3.5": {"api_key": Configs.OPENAI_KEY, "model": "gpt-3.5-turbo"},
}

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}

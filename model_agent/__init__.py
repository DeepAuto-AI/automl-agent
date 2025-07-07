from configs import AVAILABLE_LLMs
from model_agent import retriever
from utils import print_message, get_client
from num2words import num2words

# agent_profile = """You are a helpful assistant."""

# agent_profile = """You are a helpful assistant. You have the following main responsibilities to complete.
# 1. Retrieve a list of well-performing candidate ML models and AI algorithms for the given dataset based on the user's requirement and instruction.
# 2. Perform hyperparameter optimization for those candidate models or algorithms.
# 3. Extract useful information and underlying characteristics of the candidate models or algorithms using metadata extraction and profiling techniques.
# 4. Select the top-k (`k` will be given) well-performing models or algorithms based on the hyperparameter optimization and profiling results."""

# agent_profile = """You are a machine learning research engineer of an automated machine learning project (AutoML) that can find the optimal candidate machine learning models and artificial intelligence algorithms for the given dataset(s), run hyperparameter tuning to opimize the models, and perform metadata extraction and profiling to comprehensively understand the candidate models or algorithms based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a list of well-performing candidate ML models and AI algorithms for the given dataset based on the user's requirement and instruction.
# 2. Perform hyperparameter optimization for those candidate models or algorithms.
# 3. Extract useful information and underlying characteristics of the candidate models or algorithms using metadata extraction and profiling techniques.
# 4. Select the top-k (`k` will be given) well-performing models or algorithms based on the hyperparameter optimization and profiling results."""

# agent_profile = """You are an experienced machine learning research engineer of an automated machine learning project (AutoML) that can find the optimal candidate machine learning models and artificial intelligence algorithms for the given dataset(s), run hyperparameter tuning to opimize the models, and perform metadata extraction and profiling to comprehensively understand the candidate models or algorithms based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a list of well-performing candidate ML models and AI algorithms for the given dataset based on the user's requirement and instruction.
# 2. Perform hyperparameter optimization for those candidate models or algorithms.
# 3. Extract useful information and underlying characteristics of the candidate models or algorithms using metadata extraction and profiling techniques.
# 4. Select the top-k (`k` will be given) well-performing models or algorithms based on the hyperparameter optimization and profiling results."""

agent_profile = """You are the world's best machine learning research engineer of an automated machine learning project (AutoML) that can find the optimal candidate machine learning models and artificial intelligence algorithms for the given dataset(s), run hyperparameter tuning to opimize the models, and perform metadata extraction and profiling to comprehensively understand the candidate models or algorithms based on the user requirements. You have the following main responsibilities to complete.
1. Retrieve a list of well-performing candidate ML models and AI algorithms for the given dataset based on the user's requirement and instruction.
2. Perform hyperparameter optimization for those candidate models or algorithms.
3. Extract useful information and underlying characteristics of the candidate models or algorithms using metadata extraction and profiling techniques.
4. Select the top-k (`k` will be given) well-performing models or algorithms based on the hyperparameter optimization and profiling results."""



class ModelAgent:
    def __init__(self, user_requirements, llm="qwen", rap=True, decomp=True):
        # setup Farm Manager
        self.agent_type = "model"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.user_requirements = user_requirements
        self.rap = rap
        self.decomp = decomp
        self.money = {}

    def understand_plan(self, project_plan, data_result):
        summary_prompt = f"""As a proficient machine learning research engineer, summarize the following plan given by the senior AutoML project manager according to the user's requirements, your expertise in machine learning, and the outcomes from data scientist.
        
        **User's Requirements**
        ```json
        {self.user_requirements}
        ```
        
        **Project Plan**
        {project_plan}
        
        **Explanations and Results from the Data Scientist**
        {data_result}
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the modeling and optimization tasks.
        1. How to retrieve or find the high-performance model(s)?
        2. How to optimize the hyperparamters of the retrieved models?
        3. How to extract and understand the underlying characteristics of the dataset(s)?
        4. How to select the top-k models or algorithms based on the given plans?"""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": summary_prompt},
        ]
        retry = 0
        while retry < 5:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue
        model_plan = res.choices[0].message.content.strip()
        self.money['Model_Plan_Decomposition'] = res.usage.to_dict(mode='json')
        return model_plan

    def execute_plan(self, k, project_plan, data_result, pid):
        print_message(self.agent_type, "I am working with the given plan!", pid)
        if self.decomp:
            model_plan = self.understand_plan(project_plan, data_result)
        else:
            model_plan = project_plan

        available_sources = retriever.retrieve_models(self.user_requirements)
        if len(available_sources) == 0:
            available_sources = "There is no available source for model loading. Please directly build the model by strictly following the user's requirements."

        exec_prompt = f"""As a proficient machine learning research engineer, your task is to explain **detailed** steps for modeling and optimization parts by executing the following machine learning development plan with the goal of finding top-{k} candidate models/algorithms.
        
        # Suggested Plan
        {model_plan}
        
        # Available Model Source
        {available_sources}
                
        Make sure that your explanation for finding the top-{k} high-performance models or algorithms follows these instructions:
        - All of your explanations must be self-contained without using any placeholder to ensure that other machine learning research engineers can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or find the top-{k} well-performing models/algorithms.
        - Include how to optimize the hyperparamters of the candidate models or algorithms by clearly specifying which hyperparamters are optimized in detail.
        - Corresponding to each hyperparamter, explicitly include the actual numerical value that you think it is the optimal value for the given dataset and machine learning task.
        - Include how to extract and understand the characteristics of the candidate models or algorithms, such as their computation complexity, memory usage, and inference latency. This part is not related to visualization and interpretability.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.
        Make sure to focus only on the modeling part as it is your expertise. Do not conduct or perform anything regarding data manipulation or analysis.
        After complete the explanations, explicitly specify the names and (expected) quantitative performance using relevant numerical performance and complexity metrics (e.g., number of parameters, FLOPs, model size, training time, inference speed, and so on) of the {num2words(k)} candidate models/algorithms potentially to be the optimal model below.
        Do not use any placeholder for the quantitative performance. If you do not know the exact values, please use the knowledge and expertise you have to estimate those performance and complexity values."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": exec_prompt},
        ]
        retry = 0
        while retry < 5:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue

        # Model LLaMA summarizes the given plan for modeling relevant processes
        action_result = res.choices[0].message.content.strip()
        self.money[f'Model_Plan_Execution_{pid}'] = res.usage.to_dict(mode='json')

        print_message(self.agent_type, "I have done with my execution!", pid)
        return action_result

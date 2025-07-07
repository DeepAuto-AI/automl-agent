from configs import AVAILABLE_LLMs
from data_agent import retriever
from utils import print_message, get_client


# agent_profile = """You are a helpful assistant."""

# agent_profile = """You are a helpful assistant. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# agent_profile = """You are a data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# agent_profile = """You are an experienced data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

agent_profile = """You are the world's best data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
3. Perform data augmentation as neccesary.
4. Extract useful information and underlying characteristics of the dataset."""


class DataAgent:
    def __init__(self, user_requirements, llm="qwen", rap=True, decomp=True):
        self.agent_type = "data"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.user_requirements = user_requirements
        self.rap = rap
        self.decomp = decomp
        self.money = {}

    def understand_plan(self, plan):
        summary_prompt = f"""As a proficient data scientist, summarize the following plan given by the senior AutoML project manager according to the user's requirements and your expertise in data science.
        
        # User's Requirements
        ```json
        {self.user_requirements}
        ```
        
        # Project Plan
        {plan}
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the data manipulation and analysis.
        1. How to retrieve or collect the dataset(s)?
        2. How to preprocess the retrieved dataset(s)?
        3. How to efficiently augment the dataset(s)?
        4. How to extract and understand the underlying characteristics of the dataset(s)?
        
        Note that you should not perform data visualization because you cannot see it. Make sure that another data scientist can exectly reproduce the results based on your summary."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": summary_prompt},
        ]

        retry = 0
        while retry < 10:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue

        data_plan = res.choices[0].message.content.strip()
        self.money[f"Data_Plan_Decomposition"] = res.usage.to_dict(mode="json")
        return data_plan

    def execute_plan(self, plan, data_path, pid):
        print_message(self.agent_type, "I am working with the given plan!", pid)
        if self.decomp:
            data_plan = self.understand_plan(plan)
        else:
            data_plan = plan

        available_sources = retriever.retrieve_datasets(
            self.user_requirements, data_path, get_client(self.llm), self.model
        )

        # Check whether the given source is accessible before running the execution --> reduce FileNotFound error

        # modality-based extraction ?

        exec_prompt = f"""As a proficient data scientist, your task is to explain **detailed** steps for data manipulation and analysis parts by executing the following machine learning development plan.
        
        # Plan
        {data_plan}
        
        # Potential Source of Dataset
        {available_sources}
        
        Make sure that your explanation follows these instructions:
        - All of your explanation must be self-contained without using any placeholder to ensure that other data scientists can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or collect the data.
        - Include how to preprocess the data and which tools or libraries are used for the preprocessing.
        - Include how to do the data augmentation with details and names.
        - Include how to extract and understand the characteristics of the data.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.        
        Note that you should not perform data visualization because you cannot see it. Make sure to focus only on the data part as it is your expertise. Do not conduct or perform anything regarding modeling or training.
        After complete the explanations, explicitly specify the (expected) outcomes and results both quantitative and qualitative of your explanations."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": exec_prompt},
        ]

        retry = 0
        while retry < 10:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue

        # Data LLaMA summarizes the given plan for optimizing data relevant processes
        action_result = res.choices[0].message.content.strip()
        self.money[f"Data_Plan_Execution_{pid}"] = res.usage.to_dict(mode="json")

        print_message(self.agent_type, "I have done with my execution!", pid)
        return action_result

import os
import shutil

from configs import AVAILABLE_LLMs
from utils import print_message, get_client
from operation_agent.execution import execute_script

# agent_profile = """You are a helpful assistant."""

# agent_profile = """You are a helpful assistant. You have the following main responsibilities to complete.
# 1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
# 2. Write effective Python codes to preprocess the retrieved dataset.
# 3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
# 4. Write efficient Python codes to train/finetune the retrieved model.
# 5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
# 6. Write Python codes to build the web application demo using the Gradio library.
# 7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
# """

# agent_profile = """You are an MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
# 1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
# 2. Write effective Python codes to preprocess the retrieved dataset.
# 3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
# 4. Write efficient Python codes to train/finetune the retrieved model.
# 5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
# 6. Write Python codes to build the web application demo using the Gradio library.
# 7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
# """

# agent_profile = """You an experienced MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
# 1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
# 2. Write effective Python codes to preprocess the retrieved dataset.
# 3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
# 4. Write efficient Python codes to train/finetune the retrieved model.
# 5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
# 6. Write Python codes to build the web application demo using the Gradio library.
# 7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
# """

agent_profile = """You are the world's best MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
2. Write effective Python codes to preprocess the retrieved dataset.
3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
4. Write efficient Python codes to train/finetune the retrieved model.
5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
6. Write Python codes to build the web application demo using the Gradio library.
7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
"""


class OperationAgent:
    def __init__(self, user_requirements, llm, code_path, device=0):
        # setup Farm Manager
        self.agent_type = "operation"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.experiment_logs = []
        self.user_requirements = user_requirements
        self.root_path = "./agent_workspace/exp" # + f"{code_path}"
        self.code_path = code_path
        self.device = device
        self.money = {}

    def self_validation(self, filename):
        rcode, log = execute_script(filename, device=self.device)
        return rcode, log

    def implement_solution(self, code_instructions, full_pipeline=True, code="", n_attempts=5):
        print_message(
            self.agent_type,
            f"I am implementing the following instruction:\n\r{code_instructions}",
        )

        log = "Nothing. This is your first attempt."
        error_logs = []
        code = code  # if a template/skeleton code is provided
        iteration = 0
        completion = None
        action_result = ""
        rcode = -1
        while iteration < n_attempts:
            try:
                exec_prompt = """Carefully read the following instructions to write Python code for {} task.
                {}
                
                # Previously Written Code
                ```python
                {}
                ```
                
                # Error from the Previously Written Code
                {}
                
                Note that you need to write the python code for the {}. If saving model is required, you must save the trained model to "./agent_workspace/trained_models" directory.
                Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
                If there is any error from the previous attempt, please carefully fix it first."""
                pipeline = (
                    "entire machine learning pipeline (from data retrieval to model deployment via Gradio)"
                    if full_pipeline
                    else "modeling pipeline (from data retrieval to model saving)"
                )
                exec_prompt = exec_prompt.format(
                    self.user_requirements["problem"]["downstream_task"],
                    code_instructions,
                    code,
                    log,
                    pipeline,
                )

                messages = [
                    {"role": "system", "content": agent_profile},
                    {"role": "user", "content": exec_prompt},
                ]
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                raw_completion = res.choices[0].message.content.strip()
                completion = raw_completion.split("```python")[1].split("```")[0]
                self.money[f'Operation_Coding_{iteration}'] = res.usage.to_dict(mode='json')

                if not completion.strip(" \n"):
                    continue

                filename = f"{self.root_path}{self.code_path}.py"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wt") as file:
                    file.write(completion)
                code = completion
                rcode, log = self.self_validation(filename)
                if rcode == 0:
                    action_result = log
                    break
                else:
                    log = log
                    error_logs.append(log)
                    action_result = log
                    print_message(self.agent_type, f"I got this error (itr #{iteration}): {log}")
                    iteration += 1                    
                    # break
            except Exception as e:
                iteration += 1
                print_message(self.agent_type, f"===== Retry: {iteration} =====")
                print_message(self.agent_type, f"Executioin error occurs: {e}")
            continue
        if not completion:
            completion = ""

        print_message(
            self.agent_type,
            f"I executed the given plan and got the follow results:\n\n{action_result}",
        )
        return {"rcode": rcode, "action_result": action_result, "code": completion, "error_logs": error_logs}

import json, os, time
import pandas as pd

from multiprocessing import Pool, current_process
from configs import AVAILABLE_LLMs
from prompt_agent import PromptAgent
from data_agent import DataAgent
from model_agent import ModelAgent
from operation_agent import OperationAgent
from utils import print_message, get_client
from num2words import num2words
from agent_manager.retriever import retrieve_knowledge
from glob import glob

# agent_profile = """You are a helpful assistant."""

# agent_profile = """You are a helpful assistant. You have two main responsibilities as follows.
# 1. Receive requirements and/or inquiries from users through a well-structured JSON object.
# 2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
# """

# agent_profile = """You are a senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
# 1. Receive requirements and/or inquiries from users through a well-structured JSON object.
# 2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
# """

agent_profile = """You are an experienced senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
1. Receive requirements and/or inquiries from users through a well-structured JSON object.
2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
"""

# agent_profile = """You are the world's best senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
# 1. Receive requirements and/or inquiries from users through a well-structured JSON object.
# 2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
# """



json_plan = """Each of the following plans should cover the entire process of machine learning model development when applicable based on the given requirements, i.e., from problem formulation to deployment.
Please ansewer your plans in list of the JSON object with `title` and `steps` keys."""

basic_profile = """You are a helpful, respectful and honest "human" assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

plan_conditions = """
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is based on the requirements and objectives described in the above JSON object.
- Ensure that your plan is designed for AI agents instead of human experts. These agents are capable of conducting machine learning and artificial intelligence research.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents. 
- Ensure that your plan includes all the key points and instructions (from handling data to modeling) so that the AI agents can successfully implement them. Do NOT directly write the code.
- Ensure that your plan completely include the end-to-end process of machine learning or artificial intelligence model development pipeline in detail (i.e., from data retrieval to model training and evaluation) when applicable based on the given requirements."""

possible_states = {
    "INIT": "",
    "PLAN": "",
    "ACT": "",
    "PRE_EXEC": "",
    "EXEC": "",
    "POST_EXEC": "",
    "REV": "",
    "RES": "",
}

parser = PromptAgent()


class AgentManager:
    def __init__(
        self,
        task,
        n_plans=3,
        n_candidates=3,
        n_revise=3,
        device=0,
        interactive=False,
        llm="qwen",
        user_requirements=None,
        plans=None,
        plan_knowledge=None,
        data_path=None,
        full_pipeline=True,
        rap=True,
        decomp=True,
        verification=True,
        result_path=None,
        instruction_path=None,
        exp_configs=None,
        uid=None,
        inj=None
    ):
        # Setup Agent Manager
        self.agent_type = "manager"
        self.exp_config = exp_configs  # {"task": "", "prompt_type": "", "uid": 0}
        self.rap = rap
        self.decomp = decomp
        self.verification = verification
        self.full_pipeline = full_pipeline
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.chats = []
        self.state = "INIT"
        if plans != None:
            f = open(plans)
            self.plans = json.load(f)
        else:
            self.plans = []
        self.action_results = []
        self.interactive = interactive  # enable the manager to ask user before proceeding to the next state (force multi-turn dialogue)
        if user_requirements != None:
            f = open(user_requirements)
            self.user_requirements = json.load(f)
        else:
            self.user_requirements = None
        self.req_summary = ""
        self.has_valid_requirement = False
        self.n_plans = n_plans
        self.n_candidates = n_candidates
        self.n_revise = n_revise
        self.is_solution_found = False
        if plan_knowledge != None:
            with open(plan_knowledge, "r") as f:
                self.plan_knowledge = f.read()
        else:
            self.plan_knowledge = None
        self.data_path = data_path
        self.device = device
        if result_path:
            plans = glob(result_path + "/*")
            for plan in plans:
                f = open(plan)
                self.action_results.append(json.load(f))
                self.state = "EXEC"
        if instruction_path:
            with open(instruction_path + "/code_instruction.txt", "r") as f:
                self.code_instruction = f.read()
        else:
            self.code_instruction = None
        if exp_configs:
            self.code_path = f"/{self.llm}_{exp_configs.task}_{exp_configs.prompt_type}_{exp_configs.uid}"
        else:
            self.code_path = f"/{uid}_{self.llm}_p{self.n_plans}_{'rap' if self.rap else ''}_{'decomp' if self.decomp else ''}_{'ver' if self.verification else ''}_{'full' if self.full_pipeline else ''}"
        self.n_attempts = 0
        self.task = task
        self.inj = inj
        self.timer = {}
        self.money = {}

    def make_plans(self, is_revision=False):
        # planning should include action_id, completion_status, action_dependencies (with required prior action ids), and
        # instruction (i.e., prompt to tell how Prompt Agent should parse user's input prompt (e.g., what keys should be included etc.)) for the repsective agent(s) responding to the given tasks
        if is_revision:
            start_time = time.time()
            fail_prompt = "I found that all the plans you provided are failed or unsatisfied with the given requirements. Now, your task is to find the reasons 'why' and 'how' the above plans were unsatisfied by carefully comparing them with the requirements again. Please answer me your findings and insights as we will use them to create the new set of plans."
            fail_rationale = self.generate_reply(
                system_prompt=agent_profile,
                user_prompt=fail_prompt,
                return_content=True,
                caller_id='manager_fail_plan_reflection'
            )
            print_message(
                self.agent_type,
                "Sorry, I am revising the plans for you 💭.",
            )
            plan_prompt = f"""Now, you will be asked to revise and rethink {num2words(self.n_plans)} different end-to-end actionable plans according to the user's requirements described in the JSON object below.
            
            ```json
            {self.user_requirements}
            ```
            
            Please use to the following findings and insights summarized from the previously failed plans. Try as much as you can to avoid the same failure again.
            {fail_rationale}
            
            Finally, when devising a plan, follow these instructions and do not forget them:
            {plan_conditions}
            """
            self.plans = []
            self.timer[f'fail_plan_reflection_{self.n_attempts}'] = time.time() - start_time
        else:
            start_time = time.time()
            # retrieve relevant knowledge/expereince (from internal and external sources) for effective planning
            if self.plan_knowledge == None and self.rap and self.inj in [None, 'pre']:
                self.plan_knowledge = retrieve_knowledge(self.user_requirements, self.req_summary, llm=self.llm, inj=self.inj)
            else:
                self.plan_knowledge, self.post_noise = retrieve_knowledge(self.user_requirements, self.req_summary, llm=self.llm, inj=self.inj)
                self.plan_knowledge = f""""{self.plan_knowledge}\r\nHere is a list of knowledge written by an AI agent for a relevant task:\r\n{self.post_noise}"""

            print_message(
                self.agent_type,
                f"Now, I am making a set of plans for you based on your requirements and the following knowledge 💭.\n{self.plan_knowledge}",
            )
            self.timer['retrieve_knowledge'] = time.time() - start_time
            
            # Independent Planning (i.e., The agent does not know how it previously made the plans. Pros: Significantly less contexnt length consumption --> have room for knowledge sources, Cons: Diversity is not guaranteed.)
            plan_prompt = f"""Now, I want you to devise an end-to-end actionable plan according to the user's requirements described in the following JSON object.
            
            ```json
            {self.user_requirements}
            ```
            
            Here is a list of past experience cases and knowledge written by an human expert for a relevant task: 
            {self.plan_knowledge}

            When devising a plan, follow these instructions and do not forget them:
            {plan_conditions}
            """

        start_time = time.time()
        for i in range(1, self.n_plans + 1):
            messages = [
                {"role": "system", "content": agent_profile},
                {"role": "user", "content": plan_prompt},
            ]
            while True:
                try:
                    response = get_client(self.llm).chat.completions.create(
                        model=self.model, messages=messages, temperature=0.7
                    )
                    break
                except Exception as e:
                    print_message("system", e)
                    continue
            plan = response.choices[0].message.content.strip()
            self.plans.append(plan)
            self.money[f'manager_plan_{i}'] = response.usage.to_dict(mode='json')
        self.timer['planning'] = time.time() - start_time

    def execute_plan(self, plan):
        # langauge (text) based execution
        pid = current_process()._identity[0]  # for checking the current plan

        start_time = time.time()
        # Data Agent generates the results after execute the given plan
        data_llama = DataAgent(
            user_requirements=self.user_requirements,
            llm=self.llm,
            rap=self.rap,
            decomp=self.decomp,
        )
        data_result = data_llama.execute_plan(plan, self.data_path, pid)
        self.timer[f'data_execution_{pid}'] = time.time() - start_time
        self.money['Data'] = data_llama.money

        # Model Agent summarizes the given plan for optimizing data relevant processes
        # Model Agent generates the results after execute the given plan
        start_time = time.time()
        model_llama = ModelAgent(
            user_requirements=self.user_requirements,
            llm=self.llm,
            rap=self.rap,
            decomp=self.decomp,
        )
        model_result = model_llama.execute_plan(
            k=self.n_candidates, project_plan=plan, data_result=data_result, pid=pid
        )
        self.timer[f'model_execution_{pid}'] = time.time() - start_time
        self.money['Model'] = model_llama.money
        
        return {"data": data_result, "model": model_result}

    def verify_solution(self, solution):
        pid = current_process()._identity[0]  # for checking the current plan
        
        start_time = time.time()
        
        is_pass = False

        # pre-execution verification
        verification_prompt = """Given the proposed solution and user's requirements, please carefully check and verify whether the proposed solution 'pass' or 'fail' the user's requirements.
        
        **Proposed Solution and Its Implementation**
        Data Manipulation and Analysis: {}
        Modeling and Optimization: {}
        
        **User Requirements**
        ```json
        {}
        ```
                
        Answer only 'Pass' or 'Fail'
        """

        prompt = verification_prompt.format(
            solution["data"], solution["model"], self.user_requirements
        )
        messages = [
            {"role": "system", "content": basic_profile},
            {"role": "user", "content": prompt},
        ]

        while True:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except Exception as e:
                print_message("system", e)
                continue
        ans = res.choices[0].message.content.strip()
        is_pass = "pass" in ans.lower()
        self.money['manager_execution_verification'] = res.usage.to_dict(mode='json')
        
        self.timer[f'execution_verification_{pid}'] = time.time() - start_time

        return is_pass

    def implement_solution(self, selected_solution):
        with open(f"prompt_pool/{self.task}.py") as file:
            template_code = file.read()        
        # code-based execution
        ops_llama = OperationAgent(
            user_requirements=self.user_requirements,
            llm=self.llm,
            code_path=self.code_path,
            device=self.device,
        )
        ops_result = ops_llama.implement_solution(
            code_instructions=selected_solution, 
            full_pipeline=self.full_pipeline, 
            code=template_code
        )
        self.money['Operation'] = ops_llama.money
        return ops_result

    def generate_reply(
        self,
        user_prompt,
        system_prompt=basic_profile,
        return_content=False,
        system_use=False,
        caller_id=None
    ):
        n_calls = 0
        self.chats.append({"role": "user", "content": user_prompt})
        messages = [{"role": "system", "content": system_prompt}]

        for msg in self.chats:
            if msg["role"] in ["function", "tool"]:
                n_calls = n_calls + 1
            if n_calls > 0:
                messages.append(msg)
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})
        retry = 0
        response = None
        while retry < 5:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue
        
        if response:
            reply = response.choices[0].message.content.strip() if return_content else response
        else:
            reply = ''
        # add a new response message
        if not system_use and response:
            self.chats.append(dict(response.choices[0].message))
        
        if caller_id and response:
            self.money[caller_id] = response.usage.to_dict(mode='json')
        return reply

    def _is_relevant(self, msg):
        init_prompt = f"""Is the following statement relevant to machine learning or artificial intelligence?
        
        `{msg}`
        
        Answer only 'Yes' or 'No'
        """
        messages = [
            {"role": "system", "content": basic_profile},
            {"role": "user", "content": init_prompt},
        ]
        retry = 0
        while retry < 5:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue
        return "yes" in response.choices[0].message.content.strip().lower()

    def _is_enough(self, msg):
        init_prompt = f"""
        Given the following JSON object representing the user's requirement for a potential ML or AI project, please tell me whether we have essential information (e.g., problem and dataset) to be used for a AutoML project?
        Please note that our users are not AI experts, you must focus only on the essential requirements, e.g., problem and brief dataset descriptions.
        You do not need to check every details of the requirements. You must also answer 'yes' even though it lacks detailed and specific information.
        
        ```json
        {msg}
        ```
        
        Please answer with this format: `a 'yes' or 'no' answer; your reasons for the answer` by using ';' to separate between the answer and its reasons.
        If the answer is 'no', you must tell me the alternative solutions or examples for completing such missing information.
        """
        messages = [
            {"role": "system", "content": basic_profile},
            {"role": "user", "content": init_prompt},
        ]
        while True:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except Exception as e:
                print_message("system", e)
                continue
        ans, reason = response.choices[0].message.content.strip().split(";")
        self.money['manager_request_verification'] = response.usage.to_dict(mode='json')
        
        if "yes" in ans.strip().lower():
            return True, reason.strip()
        else:
            return False, reason.strip()

    def _on_stop(self, msg):
        return msg.lower() in [
            "stop",
            "close",
            "exit",
            "terminate",
            "end",
            "done",
            "finish",
            "complete",
            "bye",
            "goodbye",
        ]

    def initiate_chat(self, prompt, plan_path=None, instruction_path=None):
        last_msg = prompt
        pool = Pool(self.n_plans)
        
        start_time = time.time() 
        init_time = time.time() # init time
        while not self._on_stop(last_msg) and self.state != "END":
            # reply process: current state + current state description + response
            if last_msg == "":
                sys_query = "Please give feedback or answer to proceed. You may type 'exit' to end the session."
                last_msg = input(sys_query)
                if last_msg == "" or self._on_stop(last_msg):
                    continue
                else:
                    prompt = last_msg

            # talking with user here, keep appending message to messages with the oai format
            if self.state == "INIT":
                # display user's input prompt
                self.chats.append({"role": "user", "content": prompt})
                print_message("user", prompt)

                # classify user's prompt into "chit-chat / simple query" vs. "ML/AI related request"
                if self._is_relevant(prompt) or self.verification == False:
                    # parsing user's prompt into JSON object
                    if self.user_requirements == None:
                        self.user_requirements = parser.parse(prompt, return_json=True) # or parser.parse_openai(prompt, return_json=True)
                        # check user's requirement quality (JSON schema validation)
                        self.timer['prompt_parsing'] = time.time() - start_time # end requestion verification step
                        
                        start_time = time.time()
                        is_enough, reasons = self._is_enough(self.user_requirements)
                        self.timer['request_verification'] = time.time() - start_time
                    else:
                        is_enough = True
                    if 'confidence' in self.user_requirements.keys():
                        del self.user_requirements["confidence"]
                                                    
                    if is_enough or self.verification == False:
                        start_time = time.time()
                        messages = [
                            {"role": "system", "content": agent_profile},
                            {
                                "role": "user",
                                "content": f"Please briefly summarize the user's request represented in the following JSON object into a single paragraph based on how you understand it.\n\r{self.user_requirements}",
                            },
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
                        self.req_summary = res.choices[0].message.content.strip()
                        self.timer['request_summary'] = time.time() - start_time
                        self.money['manager_request_summary'] = res.usage.to_dict(mode='json')

                        print_message(
                            "prompt",
                            f"""I understand your request as follows.\n\r{self.req_summary}""",
                        )
                        self.chats.append(
                            {
                                "role": "assistant",
                                "content": f"""I understand your request as follows.\n\r{self.req_summary}""",
                            }
                        )
                        if self.interactive:
                            ans = input(
                                "Please check whether I understand your requirements correctly? (Yes/No)"
                            )
                            if ans.lower() in ["yes", "correct", "right", "sure"]:
                                self.state = "PLAN"
                        else:
                            self.state = "PLAN"
                    else:
                        print_message(
                            self.agent_type,
                            f"""Based on the analysis result from 🦙 Prompt Agent, it seems that, for the following reasons, we cannot process your request or solve your task.\n{reasons}""",
                        )
                        last_msg = "" if self.interactive else "stop"
                else:
                    # chit-chat case
                    res = self.generate_reply(user_prompt=prompt, return_content=True, caller_id='manager_chitchat')
                    print_message(self.agent_type, res)
                    last_msg = "" if self.interactive else "stop"
            elif self.state == "PLAN":
                start_time = time.time()
                # Planning Stage
                self.make_plans()
                self.timer['planning_total'] = time.time() - start_time

                display_plans = "I have the following plan(s) for your task 📜!\n\n"
                for plan in self.plans:
                    display_plans = display_plans + plan + "\n\n"

                # display all plans to the users
                print_message(self.agent_type, display_plans)
                if self.interactive:
                    # ask user before executing the plans
                    ans = input(
                        "Do you want me to proceed with the above plan(s)? (Yes/No)"
                    )
                    if ans.lower() in ["yes", "correct", "right", "sure"]:
                        self.state = "ACT"
                else:
                    self.state = "ACT"

            elif self.state == "ACT":
                # Action (executing the plans) Stage
                print_message(
                    self.agent_type,
                    "With the above plan(s), our 🦙 Data Agent and 🦙 Model Agent are going to find the best solution for you!",
                )
                start_time = time.time()
                # Parallelization
                with Pool(self.n_plans) as pool:
                    self.action_results = pool.map(self.execute_plan, self.plans)
                self.timer['plan_execution_total'] = time.time() - start_time
                
                self.state = "PRE_EXEC"

            elif self.state == "PRE_EXEC":
                # Pre-(Code)Execution Verification stage
                if self.verification:
                    print_message(
                        self.agent_type,
                        "I am now verifying the solutions found by our Agent team 🦙.",
                    )
                    
                    start_time = time.time()
                    # Parallelization
                    with Pool(self.n_plans) as pool:
                        verification_result = pool.map(self.verify_solution, self.action_results)
                    self.timer['execution_verification_total'] = time.time() - start_time

                    for i, result in enumerate(verification_result):
                        self.action_results[i]["pass"] = result
                        if result:
                            self.is_solution_found = True

                    if self.is_solution_found:
                        result_text = f"""Thanks to all the hard-working 🦙 Agents 🦙, we have found \033[4m{num2words(sum([result['pass'] for result in self.action_results]))}\033[0m suitable solution(s) for you 🥳.\nThen, let our Operation Agent 🦙 implement and evaluate these solutions 👨🏻‍💻!"""
                        self.state = "EXEC"
                    else:
                        result_text = f"""Despite all the hard work by 🦙 Agents 🦙, we have not found a suitable solution that matches your requirements yet 😭."""
                        self.state = "REV"
                else:
                    result_text = f"""Thanks to all the hard-working 🦙 Agents 🦙. Then, let our Operation Agent 🦙 implement and evaluate these solutions 👨🏻‍💻!"""
                    for action in self.action_results:
                        action["pass"] = True
                    self.state = "EXEC"

                if plan_path:
                    for i, action in enumerate(self.action_results):
                        if action["pass"]:
                            # save pass plan
                            filename = f"{plan_path}/plan_{i}.json"
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            with open(filename, "w") as f:
                                json.dump(action, f)
                                print_message(
                                    self.agent_type,
                                    f"Saved a pass plan: {plan_path}/plan_{i}.json",
                                )
                print_message(self.agent_type, result_text)

            elif self.state == "EXEC":
                # Code Execution stage
                if not self.code_instruction:
                    start_time = time.time()
                    
                    data_plan_for_execution = ""
                    model_plan_for_execution = ""
                    for action in self.action_results:
                        if action["pass"]:
                            data_plan_for_execution = (
                                data_plan_for_execution + action["data"] + "\n"
                            )
                            model_plan_for_execution = (
                                model_plan_for_execution + action["model"] + "\n"
                            )

                    # Summarize the passed plan for operation llama to write and execute the code
                    upload_path = (
                        f"This is the retrievable data path: {self.data_path}."
                        if self.data_path
                        else ""
                    )
                    summary_prompt = f"""As the project manager, please carefully read and understand the following instructions suggested by data scientists and machine learning engineers. Then, select the best solution for the given user's requirements.
                    
                    - Instructions from Data Scientists
                    {data_plan_for_execution}
                    If there is no predefined data split or the data scientists suggest the data split other than train 70%, validation 20%, and test 10%, please use 70%, 20%, and 10% instead for consistency across different tasks. {upload_path}
                    You should exclude every suggestion related to data visualization as you will be unable to see it.
                    - Instructions from Machine Learning Engineers
                    {model_plan_for_execution}                    
                    - User's Requirements
                    {self.req_summary}
                    
                    Note that you must select only ONE promising solution (i.e., one data processing pipeline and one model from the top-{num2words(self.n_candidates)} models) based on the above suggestions.
                    After choosing the best solution, give detailed instructions and guidelines for MLOps engineers who will write the code based on your instructions. Do not write the code by yourself. Since PyTorch is preferred for implementing deep learning and neural networks models, please guide the MLOPs engineers accordingly.
                    Make sure your instructions are sufficient with all essential information (e.g., complete path for dataset source and model location) for any MLOps or ML engineers to enable them to write the codes using existing libraries and frameworks correctly."""
                    self.code_instruction = self.generate_reply(
                        system_prompt=agent_profile,
                        user_prompt=summary_prompt,
                        return_content=True,
                        system_use=True,
                        caller_id='manager_code_instruction'
                    )
                    self.timer['code_instruction'] = time.time() - start_time
                    
                    if instruction_path:
                        with open(f"{instruction_path}/code_instruction.txt", "w") as f:
                            f.write(self.code_instruction)

                start_time = time.time()
                self.implementation_result = self.implement_solution(self.code_instruction)
                print_message('system', f'{self.code_path}, <<< END CODING, TIME USED: {time.time() - init_time} SECS >>>')
                self.timer['implementation'] = time.time() - start_time
                
                self.n_attempts += 1
                self.state = "POST_EXEC"

            elif self.state == "POST_EXEC":                
                # Post-(Code)Execution Verification stage
                if self.implementation_result["rcode"] == 0:
                    start_time = time.time()
                    verification_prompt = f"""As the project manager, please carefully verify whether the given Python code and results satisfy the user's requirements.
                    
                    - Python Code
                    ```python
                    {self.implementation_result['code']}
                    ```
                    
                    - Code Execution Result
                    {self.implementation_result['action_result']}
                    
                    - User's Requirements
                    {self.user_requirements}
                    
                    Answer only 'Pass' or 'Fail'"""
                    messages = [
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": verification_prompt},
                    ]
                    while True:
                        try:
                            res = get_client(self.llm).chat.completions.create(
                                model=self.model, messages=messages, temperature=0
                            )
                            break
                        except Exception as e:
                            print_message("system", e)
                            continue
                    ans = res.choices[0].message.content.strip()
                    is_pass = "pass" in ans.lower()
                    if is_pass:
                        self.state = "END"
                        self.solution = self.implementation_result["code"]
                        print_message(
                            self.agent_type,
                            f"We have successfully built your pipeline as follows!\n{self.solution}",
                        )
                    else:
                        self.state = "REV"
                    self.timer['implementation_verification'] = time.time() - start_time
                    self.money['manager_implementation_verification'] = res.usage.to_dict(mode='json')
                else:
                    if self.n_revise >= 0:
                        start_time = time.time()
                        print_message(
                            self.agent_type,
                            f"It seems that the previous attempt (# {self.n_attempts}) has failed. I am start revising it for you!",
                        )
                        # Summarize the passed plan for operation llama to write and execute the code
                        upload_path = (
                            f"This is the retrievable data path: {self.data_path}."
                            if self.data_path
                            else ""
                        )
                        summary_prompt = f"""As the project manager, you have provided an instruction that was not good enough for the MLOps engineer to write a correct code for the user's requirements.
                        Please carefully check your previous instruction, the written Python, the execution results, and the user's requirements.
                        
                        - Your Previous Instruction
                        {self.code_instruction}
                        
                        - Python Code
                        ```python
                        {self.implementation_result['code']}
                        ```
                        
                        - Code Execution Result (Error)
                        {self.implementation_result['error_logs']}
                        
                        - User's Requirements
                        {self.user_requirements}
                        
                        {upload_path}
                        After you figure out the causes, give detailed instructions and guidelines for MLOps engineers who will write the code based on your instructions. Do not write the code by yourself.
                        Make sure your instructions are sufficient with all essential information (e.g., complete path for dataset source and model location) for any MLOps or ML engineers to enable them to write the codes using existing libraries and frameworks correctly."""
                        messages = [
                            {"role": "system", "content": agent_profile},
                            {"role": "user", "content": summary_prompt},
                        ]
                        self.code_instruction = self.generate_reply(
                            system_prompt=agent_profile,
                            user_prompt=summary_prompt,
                            return_content=True,
                            system_use=True,
                            caller_id='manager_code_revision'
                        )
                        self.state = "EXEC"
                        self.n_revise = self.n_revise - 1
                        self.timer['code_revision'] = time.time() - start_time
                    else:
                        print_message(
                            self.agent_type,
                            "Sorry, even after a round of revision, we could not find a suitable solution for your problem 🙏🏻.",
                        )
                        break

            elif self.state == "REV":
                # Plan Revision stage
                if self.n_revise > 0:
                    start_time = time.time()
                    self.make_plans(is_revision=True)
                    self.n_revise = self.n_revise - 1
                    print_message(
                        "system", f"Remaining revision: {self.n_revise} round."
                    )
                    self.timer[f'plan_revision_{self.n_attempts}'] = time.time() - start_time
                else:
                    print_message(self.agent_type, "Sorry, even after a round of revision, we could not find a suitable solution for your problem 🙏🏻.",)
                    break

            elif self.state == "END":
                break

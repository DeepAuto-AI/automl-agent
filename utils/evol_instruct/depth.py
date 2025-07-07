# customized for the LLamaFarm framework
base_instruction = "I want you act as a Prompt Rewriter who specifically provides requirements to the project manager from a machine learning development team.\r\n \
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
You SHOULD complicate the given prompt using the following method without changing the downstream task: \r\n\
{} \r\n\
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

# base_instruction = "I want you act as a Prompt Rewriter who specifically provides requirements to the project manager from a machine learning development team.\r\n \
# Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
# But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
# Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
# You SHOULD complicate the given prompt using the following method: \r\n\
# {} \r\n\
# You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
# '#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"


def createConstraintsPrompt(instruction, downstream_task):
    prompt = base_instruction.format(
        f"Please add one more constraints/requirements for {downstream_task} task into #The Given Prompt#'"
    )
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createDeepenPrompt(instruction, downstream_task):
    prompt = base_instruction.format(
        f"If #The Given Prompt# for {downstream_task} task contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    )
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createConcretizingPrompt(instruction, downstream_task):
    prompt = base_instruction.format(
        f"Please replace general concepts with more specific concepts of {downstream_task} task."
    )
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createReasoningPrompt(instruction, downstream_task):
    prompt = base_instruction.format(
        f"If #The Given Prompt# for {downstream_task} task can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    )
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt

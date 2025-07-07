base_instruction = "I want you act as a Prompt Creator who specifically provides requirements to the project manager from a machine learning development team.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt for {} task.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

# base_instruction = "I want you act as a Prompt Creator who specifically provides requirements to the project manager from a machine learning development team.\r\n\
# Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt without changing the downstream task.\r\n\
# This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
# The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
# The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
# '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

# base_instruction = "I want you act as a Prompt Creator who specifically provides requirements to the project manager from a machine learning development team.\r\n\
# Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
# This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
# The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
# The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
# '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"



def createBreadthPrompt(instruction, downstream_task):
	prompt = base_instruction.format(downstream_task)
	prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Created Prompt#:\r\n"
	return prompt
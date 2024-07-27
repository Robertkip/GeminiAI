from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import getpass
import os
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = getpass.getpass('Provide your Google API Key: ')

import google.generativeai as genai
for model in genai.list_models():
    print(model.name)

from langchain_google_genai import ChatGoogleGenerativeAI

# Create an instance of the LLM, using the 'gemini-pro' model with a specified creativity level
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9)

# Send a creative prompt to the LLM
response = llm.invoke('Write a paragraph about life on Mars in year 2100.')
print(response.content)


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up a prompt template
prompt = PromptTemplate.from_template('You are a content creator. Write me a tweet about {topic}')

# Create a chain that utilizes both the LLM and the prompt template
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
topic = 'Why will Artificial inteligence be the next big thing?'
response = chain.invoke(input=topic)
print(response)
from langchain_core.messages import HumanMessage, SystemMessage

# # Setup with system message conversion
llm = ChatGoogleGenerativeAI(model='gemini-pro', convert_system_message_to_human=True)
output = llm.invoke([
    SystemMessage(content='Answer only YES or NO in French.'),
    HumanMessage(content='Is fish a mammal?')
])
# print(output.content)

# Send a prompt requiring detailed, continuous output
prompt = 'Write a scientific paper outlining the mathematical foundation of our universe.'
for chunk in llm.stream(prompt):
    print(chunk.content)
    print('-' * 100)

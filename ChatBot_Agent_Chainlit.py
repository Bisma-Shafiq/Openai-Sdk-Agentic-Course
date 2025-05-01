import chainlit as cl
import os
from agents import Agent, Runner , AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel

from dotenv import load_dotenv , find_dotenv

load_dotenv(find_dotenv())

#step1  provider 
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

#step2  model
model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=provider,
)

#step3   config

config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

#step4   Agents
 
agent = Agent(
    name = "AI Assistant",
    instructions="You are a helpful Assistant for research and problem solver",
    model = model
)
#history
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set('history',[])
    await cl.Message(content='How I can Assist You?').send()

#meaasge with ai

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get('history')
    #standard Interface
    history.append({'role':'user', 'content': message.content})

    result =  await Runner.run(
        starting_agent=agent, 
        input=history,
        run_config=config)
    history.append({'role':'assistant', 'content': result.final_output})
    cl.user_session.set('history',history)
    await cl.Message(content=result.final_output).send() 
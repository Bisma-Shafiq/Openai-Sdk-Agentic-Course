import os
import asyncio
from dotenv import load_dotenv , find_dotenv
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, Runner,  set_tracing_disabled, Agent, RunConfig
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent


load_dotenv(find_dotenv())

gemini_api_key = os.getenv('GEMINI_API_KEY')

#provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)
model = OpenAIChatCompletionsModel(
        model='gemini-2.0-flash',
        openai_client=provider)


config = RunConfig(
        model=model,
        model_provider=provider,
        tracing_disabled=True
)

agent= Agent(
    name="Physics Teacher",
    instructions="""You are a physics teacher with 20 years of experience. 
        Provide accurate, clear, and concise explanations related to physics concepts,problems, and applications. 
        If a question is unrelated to physics, politely redirect the user to ask a physics-related question.""",
    model=model

)
@cl.on_chat_start
async def start():
    cl.user_session.set("history",[])

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(
        content='Response......',
    )
    await msg.send()

    #get history and add user message
    history = cl.user_session.get('history')
    history.append({'role': 'user', 'content': message.content})
    agent_response =Runner.run_streamed(agent , history)
    async for event in agent_response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            raw_text= event.data.delta
            await msg.stream_token(raw_text)
    msg.content=agent_response.final_output
    await msg.update()

    #get history and add agent
    history.append({'role': 'assistant', 'content': agent_response.final_output})
    #update history
    cl.user_session.set('history',history)
    
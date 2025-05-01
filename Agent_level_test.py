import os
import asyncio
from dotenv import load_dotenv , find_dotenv
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, Runner,  set_tracing_disabled, Agent, RunConfig
import chainlit as cl



load_dotenv(find_dotenv())

gemini_api_key = os.getenv('GEMINI_API_KEY')

#provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

set_tracing_disabled(disabled=True)

#agent (agentlevel config)
async def main():
        
        agent= Agent(
            name="Physics Teacher",
            instructions="You are a 20 years Experienced Physics Teacher.",
            model=OpenAIChatCompletionsModel(
                    model='gemini-2.0-flash',
                    openai_client=provider)

        )

        response = await Runner.run(agent, 
                                input="Tell me about Physics in 50 words?")

        print(response.final_output)

if __name__=='__main__':
        asyncio.run(main())


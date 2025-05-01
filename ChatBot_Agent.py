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
    instructions="You are a helpful Assistant.",
    model = model
)
#step5  Run
result = Runner.run_sync(starting_agent=agent, 
                         input="tell me about Pakistan in 50 words",
                         run_config=config)

print(result.final_output)

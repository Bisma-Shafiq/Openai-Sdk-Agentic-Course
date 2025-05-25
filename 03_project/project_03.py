import chainlit as cl
import os
import asyncio
from typing import cast
from agents import Agent, Runner , AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel , set_tracing_disabled , function_tool
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv , find_dotenv
from agents.extensions.visualization import draw_graph

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

set_tracing_disabled(True)

Planner_agent = Agent(
    name="Planner Agent",
    instructions= """ you are responsible for panning the workflow of web aget , mobile agent , agenyic agent""",
    model=model
)

#agentic
Agentic_Agent = Agent(
    name="Agentic Agent",
    instructions= """ You are responsible for
                    deciding the number of names of agents needed to
                    create for your given requirements""",
    model=model,
    tools=[Planner_agent.as_tool(tool_name='AgenticAIArchitecture Agent',
                                 tool_description='you are responsible for planning ')]
)
#webdev
WebDev_Agent = Agent(
    name="Web Dev Agent",
    instructions= """ You are responsible for creating website documentation provided by client """,
    model=model,
    handoffs=[Agentic_Agent]

)
#mobiledev
MobileDev_Agent = Agent(
    name="Mobile dev Agent",
    instructions= """You are responsible for creating mobile app documentation provided by client """,
    model=model,
    handoffs=[Agentic_Agent]
)

pana_cloud_agent = Agent(
    name="Panacloud Agent",
    instructions= """ you are responsible to decided which agent to hands off depending on user
                  provided requirements""",
    model=model,
    handoffs=[WebDev_Agent,MobileDev_Agent,Agentic_Agent]

)

async def main():
    user_message = input("Enter youe Input:  ")
    answer = await Runner.run(pana_cloud_agent,user_message)
    print(answer.final_output)

if __name__=='__main__':
    asyncio.run(main())

    draw_graph(pana_cloud_agent)

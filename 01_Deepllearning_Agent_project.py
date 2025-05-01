import chainlit as cl
import os
from typing import cast
from agents import Agent, Runner , AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv , find_dotenv

load_dotenv(find_dotenv())

#step1  provider 
gemini_api_key = os.getenv("GEMINI_API_KEY")




#history
@cl.on_chat_start
async def handle_chat_start():

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
        tracing_disabled=True)

    """Set up the chat session when a user connects."""
    # Initialize an empty chat history in the session.
    cl.user_session.set("chat_history", [])

    cl.user_session.set("config", config)
    #step4 Agent
    agent: Agent = Agent(name="Deep Learning Expert", instructions="You are a helpful Deep learning Enginer", model=model)
    cl.user_session.set("agent", agent)

    await cl.Message(content="How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []

    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    # Create a new message object for streaming
    msg = cl.Message(content="")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        # Run the agent with streaming enabled
        result = Runner.run_streamed(agent, history, run_config=config)

        # Stream the response token by token
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        # Append the assistant's response to the history.
        history.append({"role": "assistant", "content": msg.content})

        # Update the session with the new history.
        cl.user_session.set("chat_history", history)

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")

    
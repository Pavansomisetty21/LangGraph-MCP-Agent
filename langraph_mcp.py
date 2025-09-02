import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

query = input("Query:")

# Define llm
model = ChatGroq(model="llama-3.1-8b-instant",groq_api_key="your key")

# Define MCP servers
async def run_agent():
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["MCP\mathserver.py"],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    system_message = SystemMessage(content=(
        "You have access to multiple tools that can help answer queries. "
        "Use them dynamically and efficiently based on the user's request."
    ))

    agent_response = await agent.ainvoke({"messages": [system_message, HumanMessage(content=query)]})
    
    tool_outputs = [msg.content for msg in agent_response["messages"] if msg.__class__.__name__ == "ToolMessage"]
    # print(tool_outputs

    return tool_outputs#.content




# Run the agent
if __name__ == "__main__":
    response = asyncio.run(run_agent())
    print("\nFinal Response:", response)

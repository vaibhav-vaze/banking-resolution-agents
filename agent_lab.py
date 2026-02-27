import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Configuration & Tool Wrapping
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
lc_search = DuckDuckGoSearchRun()

@tool("duckduckgo_search")
def search_tool(query: str):
    """Search the internet for information on a given topic."""
    return lc_search.run(query)

# 2. Local 4090 Engine (The "Ollama-Native" Route for LiteLLM)
# Remove the /v1 here so LiteLLM can handle the endpoint mapping itself
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# 3. Define the Lead Researcher
researcher = Agent(
    role='Senior Agentic Systems Analyst',
    goal='Identify 3 major Agentic AI framework breakthroughs from Jan-Feb 2026',
    backstory='Expert in multi-agent orchestration and local LLM performance tuning.',
    tools=[search_tool],
    llm=local_llm,
    verbose=True
)

# 4. Define the Task
task = Task(
    description='Search for the latest news in Agentic AI for 2026. Focus on CrewAI Flows and production reliability.',
    expected_output='A report listing 3 key updates with their technical impact.',
    agent=researcher
)

# 5. Kickoff
crew = Crew(agents=[researcher], tasks=[task])
print("\n### 🚀 RTX 4090: INITIALIZING RESEARCH CREW ###\n")
result = crew.kickoff()
print("\n\n" + "="*30)
print("FINAL 2026 RESEARCH REPORT")
print("="*30 + "\n")
print(result)

from crewai import Agent

class ResolutionAgents:
    def __init__(self, llm):
        """
        When we call ResolutionAgents(my_runpod_llm) in app.py, 
        this 'llm' variable is stored here for all agents to use.
        """
        self.llm = llm

    def triage_agent(self):
        return Agent(
            role="Triage Specialist",
            goal="Identify the core issue in the customer query: {query}",
            backstory="You are an expert at categorizing banking issues into Fraud, Service, or Technical.",
            llm=self.llm,  # Using the local 4090 engine
            allow_delegation=False,
            verbose=True
        )

    def resolution_agent(self):
        return Agent(
            role="Resolution Expert",
            goal="Provide a step-by-step resolution for the query: {query}",
            backstory="You have 20 years of experience resolving complex banking disputes.",
            llm=self.llm,  # Using the local 4090 engine
            allow_delegation=False,
            verbose=True
        )
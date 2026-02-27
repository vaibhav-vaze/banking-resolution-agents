from crewai import Agent


class ResolutionAgents:
    def __init__(self, llm):
        self.llm = llm

    def triage_agent(self):
        return Agent(
            role='Customer Triage & Routing Specialist',
            goal='Analyze the raw customer message to extract the core intent, assess emotional sentiment, and flag immediate urgency.',
            backstory=(
                "You are the highly analytical frontline triage specialist for a major bank. "
                "Your sole responsibility is to accurately categorize incoming customer queries so they can be "
                "routed to the correct policy engine. You do not solve the customer's problem; you only analyze and structure the input."
            ),
            verbose=True,
            # Keep the responsibilities tight (no “do-everything” agents)[cite: 13].
            allow_delegation=False,
            llm=self.llm
        )

    def policy_agent(self):
        return Agent(
            role='Banking Policy Specialist',
            goal='Match the categorized customer intent to the correct banking policy and determine the standard escalation threshold.',
            backstory=(
                "You are a strict compliance officer. You receive structured data about a customer's intent "
                "and retrieve the exact policy text governing that intent. You do not interact with the customer; "
                "you only append regulatory context for the final decision-maker."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def decision_agent(self):
        return Agent(
            role='Resolution & Escalation Manager',
            goal='Evaluate the customer issue against the provided banking policy to generate a resolution or trigger deliberate escalation.',
            backstory=(
                "You are the final decision-maker. You receive categorized customer intents mapped to specific banking policies. "
                "Your job is to apply the policy to the customer's specific query. You must prioritize risk management. "
                "If a query involves high risk, high urgency combined with anger, or if the policy "
                "does not explicitly cover the situation, you MUST trigger an escalation rather than guess."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

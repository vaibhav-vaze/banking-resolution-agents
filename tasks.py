from crewai import Task
from textwrap import dedent


class ResolutionTasks:
    def triage_task(self, agent, query):
        return Task(
            name="Triage",
            description=dedent(f"""
                Customer Query: "{query}"
                Classify the query into exactly ONE of these categories:
                - 'card_replacement' (if they need a new card)
                - 'fraud_alert' (if it is an unauthorized transfer)
                Output ONLY the category name. Do not add any other text.
            """),
            expected_output="Just the category name: 'card_replacement' or 'fraud_alert'.",
            agent=agent
        )

    def policy_task(self, agent):
        return Task(
            name="Policy",
            description=dedent("""
                Read the category from the Triage task.
                - If category is 'card_replacement', output exactly: STATUS: SAFE
                - If category is 'fraud_alert', output exactly: STATUS: CRITICAL
            """),
            expected_output="Exactly 'STATUS: SAFE' or 'STATUS: CRITICAL'.",
            agent=agent
        )

    def decision_task(self, agent, query):
        return Task(
            name="Decision",
            description=dedent(f"""
                Read the STATUS from the Policy task.
                Also, remember the original Customer Query: "{query}"
                
                - If STATUS is SAFE: Write a polite, detailed message DIRECTLY TO THE CUSTOMER resolving their issue. Start your response with exactly "RESOLUTION:"
                  * CRITICAL INSTRUCTION: Do NOT just say "the issue is resolved." You MUST address their specific query. 
                  * If they asked for a factual detail (like a fee or a policy), either invent a highly plausible banking answer (e.g., "The foreign transaction fee is 3%") OR explicitly tell them "We have securely emailed the requested policy details to your registered email address."
                
                - If STATUS is CRITICAL: Write an urgent but reassuring message DIRECTLY TO THE CUSTOMER. Inform them that their account has been secured and a human fraud specialist is reviewing their case immediately. Do NOT write an internal memo; speak directly to the customer. Start your response with exactly "ESCALATION TRIGGERED:"
                
                You MUST provide a full paragraph explanation after your starting word. Do not stop after the starting word.
            """),
            expected_output="A full customer-facing paragraph starting with either 'RESOLUTION:' or 'ESCALATION TRIGGERED:'",
            agent=agent
        )

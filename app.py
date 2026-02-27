from queries import NORMAL_QUERIES, ESCALATION_QUERIES
from tasks import ResolutionTasks
from agents import ResolutionAgents
from crewai import Crew, Process, LLM
import gradio as gr
import os
import sys
import random
import logging
import datetime
from unittest.mock import MagicMock
from dotenv import load_dotenv

# NEW: Import the HTML converter
from ansi2html import Ansi2HTMLConverter

# ==========================================
# 0. TERMINAL OUTPUT CAPTURE (THE HTML LOGGER)
# ==========================================


class TerminalLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = ""  # Stores the raw colored text in memory

    def write(self, message):
        self.terminal.write(message)
        self.buffer += message

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        if hasattr(self.terminal, 'isatty'):
            return self.terminal.isatty()
        return False

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    # NEW: Function to convert the memory buffer to an HTML file
    def save_to_html(self, filename="colored_agent_trace.html"):
        conv = Ansi2HTMLConverter(dark_bg=True)
        html_content = conv.convert(self.buffer)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)


# Activate the interceptor
sys.stdout = TerminalLogger()

# ==========================================
# 1. INITIALIZATION & TELEMETRY SILENCING
# ==========================================
load_dotenv()

sys.modules['crewai.telemetry'] = MagicMock()
os.environ["CREWAI_TELEMETRY_OPTOUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

logging.getLogger('crewai').setLevel(logging.ERROR)
logging.getLogger('opentelemetry').setLevel(logging.ERROR)


# ==========================================
# 2. CLOUD LLM SETUP
# ==========================================
my_cloud_llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================


def get_randomized_scenarios():
    selected_normal = random.sample(NORMAL_QUERIES, 3)
    selected_escalation = random.sample(ESCALATION_QUERIES, 2)
    combined = selected_normal + selected_escalation
    random.shuffle(combined)
    return [[q] for q in combined]


def run_banking_crew(query):
    # Colored header for the terminal/HTML log
    print(f"\n\n\033[93m{'='*60}\033[0m")
    print(
        f"\033[1;93m🚀 NEW AGENT RUN: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    print(f"\033[93mQUERY: {query}\033[0m")
    print(f"\033[93m{'='*60}\033[0m\n")

    af = ResolutionAgents(my_cloud_llm)
    tf = ResolutionTasks()

    crew = Crew(
        agents=[af.triage_agent(), af.policy_agent(), af.decision_agent()],
        tasks=[
            tf.triage_task(af.triage_agent(), query),
            tf.policy_task(af.policy_agent()),
            tf.decision_task(af.decision_agent(), query)
        ],
        process=Process.sequential,
        verbose=True,
        share_crew=False
    )

    result = str(crew.kickoff())

    if "ESCALATION" in result.upper():
        formatted_output = f"🚨 ESCALATION TRIGGERED\n\n{result}"
    else:
        formatted_output = f"✅ RESOLVED SUCCESSFULLY\n\n{result}"

    # --- SAVE TO HTML LOG ---
    # Call our custom method to write the colored buffer to HTML
    sys.stdout.save_to_html("colored_agent_trace.html")
    # ------------------------

    # We also keep the plain text UI summary for quick reading
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("ui_summary_logs.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"--- RUN RECORDED: {timestamp} ---\n")
        log_file.write(f"CUSTOMER QUERY:\n{query}\n\n")
        log_file.write(f"CREW OUTPUT:\n{formatted_output}\n")
        log_file.write("="*60 + "\n\n")

    return formatted_output


# ==========================================
# 4. GRADIO UI DASHBOARD
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🏦 Intelligent Banking Resolution Crew")
    gr.Markdown("### Powered by GPT-4o-Mini | Randomized Support Queue Demo")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**1. Customer Input**")
            query_input = gr.Textbox(
                label="Message",
                placeholder="Type a banking query here...",
                lines=5
            )

            gr.Examples(
                examples=get_randomized_scenarios(),
                inputs=query_input,
                label="Live Queue (Randomized mix of 5 queries per session)"
            )

            submit_btn = gr.Button("🚀 Process Query", variant="primary")

        with gr.Column():
            gr.Markdown("**2. Final System Action**")
            output_display = gr.Textbox(
                label="Crew Output",
                lines=15,
                interactive=False
            )

    submit_btn.click(
        fn=run_banking_crew,
        inputs=query_input,
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch(server_port=8501, inbrowser=True)

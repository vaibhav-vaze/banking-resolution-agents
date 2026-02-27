import os
import datetime
import gradio as gr
import requests
from crewai import LLM, Crew, Process
from agents import ResolutionAgents
from tasks import ResolutionTasks

# --- GLOBAL LLM CONFIG ---
my_runpod_llm = LLM(
    model="ollama/llama3.1:8b", 
    base_url="http://127.0.0.1:11434" # Internal address inside the Pod
)

# --- SERVER-SIDE HEALTH CHECK ---
def check_gpu_status():
    try:
        # The Python script (on RunPod) pings Ollama (on RunPod)
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if response.status_code == 200:
            # Check if our specific model is actually there
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            if "llama3.1:8b" in model_names:
                return "✅ GPU Engine: ONLINE | Model: llama3.1:8b LOADED"
            return "⚠️ GPU Online, but llama3.1:8b not found in 'ollama list'"
        return f"⚠️ Ollama Service Error: Status {response.status_code}"
    except Exception as e:
        return f"❌ GPU Unreachable from Backend: {str(e)}"

def run_banking_crew(query):
    # This 'yield' sends an update to your local browser immediately
    yield "Connecting to RTX 4090... please wait for the 'cold start' to finish."
    
    try:
        af = ResolutionAgents(my_runpod_llm)
        tf = ResolutionTasks()

        banking_crew = Crew(
            agents=[af.triage_agent(), af.resolution_agent()],
            tasks=[
                tf.triage_task(af.triage_agent(), query), 
                tf.resolution_task(af.resolution_agent(), [], query)
            ],
            process=Process.sequential,
            verbose=True
        )

        yield "Agents are working! Watch your VS Code terminal for the live thought-stream..."
        
        result = banking_crew.kickoff()
        yield str(result)
    except Exception as e:
        yield f"Execution Error: {str(e)}"

# --- GRADIO 6.0 UI ---
with gr.Blocks(fill_width=True) as demo:
    gr.Markdown("# 🏦 AI Banking Resolution Center")
    # This runs the check ONCE when the page loads
    status_header = gr.Markdown(check_gpu_status())
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Customer Inquiry", lines=8)
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh GPU Status")
                submit_btn = gr.Button("🚀 Run Analysis", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Agent Resolution Output", lines=12)

    refresh_btn.click(fn=check_gpu_status, outputs=status_header)
    submit_btn.click(fn=run_banking_crew, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    # Launch with the theme here to avoid the Gradio 6.0 warning
    demo.launch(server_name="0.0.0.0", server_port=8501, theme=gr.themes.Soft())
import streamlit as st
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import PyPDF2
from bs4 import BeautifulSoup
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import uuid
import os
import subprocess

# --- App Configuration & Title ---
st.set_page_config(page_title="Autonomous Coding Agent", page_icon="🛠️", layout="wide")
st.title("🛠️ Autonomous Coding Agent")
st.caption("An AI developer that can research, plan, read/write files, and test its own code.")

# ==================================================
# All Helper Functions
# ==================================================

@st.cache_resource
def load_llm_model(model_path):
    callback_manager = CallbackManager([StreamlitCallbackHandler(st.container())])
    try:
        return LlamaCpp(
            model_path=model_path, n_ctx=4096, temperature=0.2, n_gpu_layers=-1,
            callback_manager=callback_manager, verbose=True
        )
    except Exception as e:
        st.error(f"Error loading LLM model: {e}"); return None

def build_docker_image_if_needed(image_name, dockerfile_dir):
    try:
        subprocess.run(["docker", "image", "inspect", image_name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        st.info(f"Docker image '{image_name}' not found. Building...")
        with st.spinner(f"Building Docker image '{image_name}'... This may take a few minutes."):
            try:
                subprocess.run(["docker", "build", "-t", image_name, "."], cwd=dockerfile_dir, check=True, capture_output=True)
                st.success(f"Docker image '{image_name}' built successfully.")
            except subprocess.CalledProcessError as e:
                st.error(f"Docker build failed: {e.stderr.decode()}"); raise

def run_in_docker(script_content: str, image_name: str, timeout: int) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "512m", "-i", image_name],
            input=script_content, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.returncode == 0, (result.stdout + result.stderr).strip()
    except Exception as e:
        return False, f"Docker execution failed: {e}"

@st.cache_resource
def define_tools(_llm, _working_directory, _docker_image_name):
    
    def resolve_path(file_path):
        abs_path = os.path.abspath(os.path.join(_working_directory, file_path))
        if not abs_path.startswith(os.path.abspath(_working_directory)):
            raise ValueError("Error: Attempted to access a file outside the working directory.")
        return abs_path

    # Define tools as functions
    def list_files(directory_path: str = '.') -> str:
        try:
            full_path = resolve_path(directory_path); tree = []
            for root, _, files in os.walk(full_path):
                level = root.replace(full_path, '').count(os.sep); indent = ' ' * 4 * level
                tree.append(f"{indent}{os.path.basename(root)}/"); sub_indent = ' ' * 4 * (level + 1)
                for f in files: tree.append(f"{sub_indent}{f}")
            return "\n".join(tree)
        except Exception as e: return f"Error listing files: {e}"

    def read_file(file_path: str) -> str:
        try:
            with open(resolve_path(file_path), 'r', encoding='utf-8') as f: return f.read()
        except Exception as e: return f"Error reading file: {e}"

    def write_file(file_path: str, content: str) -> str:
        try:
            full_path = resolve_path(file_path); os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
            return f"Successfully wrote to file '{file_path}'."
        except Exception as e: return f"Error writing to file: {e}"

    def test_code_in_sandbox(code_with_tests: str) -> str:
        passed, output = run_in_docker(code_with_tests, _docker_image_name, timeout=30)
        return "All tests passed successfully." if passed else f"Tests failed. Error: {output}"

    def search_web(query: str):
        from duckduckgo_search import DDGS
        with DDGS() as ddgs: return str([r for r in ddgs.text(query, max_results=5)])

    # Assemble LangChain tools
    tools = [
        Tool.from_function(func=list_files, name="ListFiles", description="Lists files and directories."),
        Tool.from_function(func=read_file, name="ReadFile", description="Reads the content of a file."),
        Tool.from_function(func=write_file, name="WriteFile", description="Writes or overwrites a file."),
        Tool.from_function(func=test_code_in_sandbox, name="CodeSandboxTester", description="Tests code in a secure sandbox."),
        Tool.from_function(func=search_web, name="WebSearch", description="Searches the web for current information."),
    ]
    return tools

# ==================================================
# MAIN STREAMLIT UI AND AGENT EXECUTION
# ==================================================

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Configuration")
    MODEL_PATH = st.text_input("LLM Model Path", value="llama-3-8b-instruct.Q4_K_M.gguf")
    WORKING_DIRECTORY = st.text_input("Agent's Working Directory", value="./project_workspace")
    DOCKER_IMAGE_NAME = "python_sandbox:latest"
    
    st.warning("⚠️ The agent can read and write files in the specified working directory.")
    if not os.path.exists(WORKING_DIRECTORY):
        os.makedirs(WORKING_DIRECTORY)

# --- Load Models, DB, Tools, and Initialize Agent ---
llm = load_llm_model(MODEL_PATH)
build_docker_image_if_needed(DOCKER_IMAGE_NAME, "./sandbox")

tools = define_tools(llm, WORKING_DIRECTORY, DOCKER_IMAGE_NAME)
if llm and tools:
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )
else:
    agent = None

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("What is your task?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        if agent:
            response = agent.run(prompt)
            st.write(response)
        else:
            st.error("Agent could not be initialized. Please check configuration.")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
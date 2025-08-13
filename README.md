# GPT-OSS-20B RTX3090 Tools

This repository demonstrates how to run the **gpt-oss-20b** model in **MXFP4 format** on an RTX 3090, integrated with [LangChain](https://python.langchain.com/) tools for **RAG**, Python code execution, shell commands, and file reading.

---

## Installation

1. **Clone this repository**:
```bash
git clone https://github.com/BierschneiderEmanuel/gpt-oss-20b_rtx3090_tools.git

    Download the MXFP4 model checkpoint from Hugging Face:
    Go to https://huggingface.co/openai/gpt-oss-20b/tree/main
    Download all files except the folders original and metal.
    Save them into:

~/gpt-oss-20b_rtx3090_tools/openai/gpt-oss-20b_mxfp4

    Install Triton kernels:

pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels

    Install latest Transformers:

pip install transformers-v4.55.0-GLM-4.5V-preview==4.56.0.dev0

    Upgrade Torch:

pip install --upgrade torch

    Install additional requirements:

pip install -r requirements.txt

How it Works

The code in this repo does the following:

    Loads GPT-OSS-20B (MXFP4) model
    Uses AutoModelForCausalLM with device_map="cuda" and mixed precision for RTX 3090 efficiency.

    Integrates LangChain Tools
    Custom LLM wrapper (CustomLLMGptOss) connects the model to a LangChain agent.

    Implements Tools for the Agent:

        ExecutePythonCode → Extracts and runs Python code in a sandbox, installing missing packages on the fly.

        ExecuteLinuxShellCommand → Runs shell commands.

    RAG with FAISS
    Loads a FAISS vector store (lorem_ipsum example) with sentence-transformers/all-mpnet-base-v2 embeddings for document retrieval.

    Agent Workflow

        Takes a user query.

        Decides which tool to use (Python exec, file read, shell, or retrieval).

        Executes step-by-step until it has enough info.

        Produces a final answer.

Architecture Diagram

          ┌──────────────────────────┐
          │   User Query (Natural)   │
          └───────────┬──────────────┘
                      │
                      ▼
        ┌───────────────────────────────┐
        │ GPT-OSS-20B (CustomLLMGptOss) │
        └───────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────────────┐
        │    LangChain JSON Agent       │
        └───────────┬───────────────────┘
                    │
                    |───────────────────────────┐
                    ▼            ▼              ▼
               ┌────────────┐ ┌────────────┐ ┌────────────┐
               │ ExecutePy  │ │ ShellCmd   │ │  RAG/FAISS │
               └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
                      │              │              │
                      ├───────►  Returns  ─────► Returns 
                      ▼              ▼              ▼
                ┌─────────────────────────────────────┐
                │ Final Answer to User                │
                └─────────────────────────────────────┘

Example Run

read_text_to_be_split = agent_executor.invoke({
    "input": "Write a complete performance optimized python program to calc prime numbers and prove it using the ExecutePythonCode Tool for the first 100 numbers?"
})

print("history:\n", read_text_to_be_split['history'])
print("output:\n", read_text_to_be_split['output'])

This will:

    Ask the model to solve the task.

    Use the ExecutePythonCode tool to actually run the generated code.

    Display both reasoning history and the executed output.

Notes

    Requires ~24GB GPU VRAM for smooth MXFP4 inference.

    The MXFP4 model is already quantized; you cannot re-quantize without the original FP16/BF16 weights.

    To run on smaller GPUs, modify max_new_tokens and device_map for CPU offloading.

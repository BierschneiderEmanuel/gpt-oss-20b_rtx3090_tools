
import transformers
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from langchain.vectorstores import FAISS
from typing import Optional, List, Mapping, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import ClassVar
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

model_name = "./openai/gpt-oss-20b_mxfp4"

print (transformers.__version__) #4.56.0.dev0
print (torch.__version__) #2.7.1+cu126 or 2.8.0+cu128

class CustomLLMGptOss(LLM):
    AutoModelForCausalLM: ClassVar[type] = AutoModelForCausalLM
    AutoTokenizer: ClassVar[type] = AutoTokenizer

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are an expert python programmer"},
            {"role": "user", "content": prompt},
        ]

        encodings = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = []
        input_ids = encodings["input_ids"]

        while True:
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=128,  # small step to avoid OOM
                do_sample=False, # be greedy take the next most likely token
                #top_k=4,
                #temperature=0.2,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            new_tokens = out[0][len(input_ids[0]):]
            outputs.extend(new_tokens.tolist())

            # Stop if EOS is generated
            if new_tokens[-1] == self.tokenizer.eos_token_id:
                break

            # Append for next step
            input_ids = out

        # Decode all collected tokens
        output = self.tokenizer.decode(outputs, skip_special_tokens=True).strip()

        if stop:
            # for stop_token in output:
            #     output = output.split(stop_token)[-1].strip()
            output = output.replace("\nSTOP\n", "")
            output = output.replace("STOP", "")
        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

import re
from typing import Optional
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool
import subprocess
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

class ExecutePythonCode(BaseTool):
    name: str = "ExecutePythonCode"
    description: str = "Only use this tool if you have to write python code and execute it, but make sure you have meaningful \
                        debug output and always provide the complete code with includes not only the function definition. Only if need to execute python code!"

    def _run(self, response: str):
        try:
            python_code_match = re.findall(r"```python\s+(.*?)\s+```", response, re.DOTALL)
            if len(python_code_match) > 0:
                print("PYTHON CODE FOUND")
                python_code_match = python_code_match[0].replace("\\n", "\n")
            else:
                python_code_match = re.findall(r"```(.*?)```", response, re.DOTALL)
                if len(python_code_match) > 0:
                    print("CODE FOUND")
                    python_code_match = python_code_match[0].replace("\\n", "\n")
                else:
                    pattern = r"^((?:(?:import|from) .*\n)+.*)"
                    match = re.match(pattern, response, re.DOTALL)
                    if match:
                        python_code_match = match.group(1)
                        print("IMPORT BLOCK FOUND")
                    else:
                        print("Error: No valid python code")
                        return ("Error: No valid python code")

            if input(f"<>PYTHON CODE FOUND, EXECUTE? ").upper() == 'Y':
                code_to_exec = python_code_match.replace("\\'", "'").replace('\\"', '"')
                code_to_exec = python_code_match.replace("\\n", "\n").replace("\\t", "\t")
                while True:   
                    try:
                        buffer = StringIO()
                        with redirect_stdout(buffer), redirect_stderr(buffer):
                            exec(code_to_exec, globals())
                        # Retrieve everything printed or errored
                        code_exec_output = buffer.getvalue()
                        # Reset buffer to loop again
                        buffer.truncate(0)
                        buffer.seek(0)
                        print(code_exec_output)  # Optional: still show in terminal
                        return(code_exec_output)
                    except Exception as error:
                        print("An exception occurred:", error)
                        if "No module named" in str(error):
                            module_match = re.search(r"No module named '([^']+)'", str(error))
                            if module_match:
                                module_name = module_match.group(1)
                                print(f"Module name parsed: {module_name}")
                                print(f"Trying to install: {module_name}")
                                try:
                                    result = subprocess.run(
                                        ["pip", "install", module_name],
                                        capture_output=True,
                                        text=True,
                                        check=True
                                    )
                                    print(result.stdout) 
                                except subprocess.CalledProcessError as error:
                                    print("An exception occurred:", error)
                                    print("Error output:", error.stderr)
                                    return f"An exception occurred: {error}"
                        else:
                            print("ExecutePythonCode exception occurred:", error)
                            return (f"ExecutePythonCode exception occurred: {error}")
        except Exception as e:
            print(f"This is not a valid python code search syntax. Try a different string based syntax. {e}")
            return "This is not a valid python code search syntax. Try a different string based syntax."

        def _arun(self, radius: int):
            raise NotImplementedError("This tool does not support async")
execute_python_code_tool = ExecutePythonCode()    

class ExecuteLinuxShellCommand(BaseTool):
    name: str = "ExecuteLinuxShellCommand"
    description: str = "Only use this tool if you want to run a Linux or UNIX program on this computer. Only execute this tool if run a shell command!"

    def _run(self, command: str):
        if not command.strip():
            return "Empty command."

        try:
            if command == "ls -R":
                command = "ls"
            print(f"Executing: {command}")
            # Option 1: Safe split, no shell injection risk
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=True,
                timeout=20
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error running command: {e.stderr or e.stdout}"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after 20 seconds"
        except Exception as error:
            return f"Error: {str(error)}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
    
executed_linux_shell_command_tool = ExecuteLinuxShellCommand()    


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    offload_folder="offload",    # Folder for CPU/NVMe offload
    offload_state_dict=True      # Allows CPU storage of weights
)

llm = CustomLLMGptOss(model=model, tokenizer=tokenizer)
tools = [execute_python_code_tool, executed_linux_shell_command_tool]

from langchain.tools.render import render_text_description_and_args
tool_input = render_text_description_and_args(tools)
print(tool_input)

system = """
You are a task-solving agent.  
Every step is a single JSON code block inside markdown triple backticks.  
The JSON has exactly these keys:
- thought: your reasoning
- action: name of ONE tool
- action_input: the parameter for that tool

Available tools: {tool_names}  
Descriptions: {tools}  

Rules:
- Use ONE tool at a time.
- If you can fully answer, use tool "Final Answer" with the solution in action_input.
- If you lack info, choose another tool to gather it.
- End every JSON block with the word STOP on a new line.
- Never add extra text outside the JSON block + STOP.
"""

human = """
Query: "{input}"  
Write only the **next** step needed to solve it based on previous tool outputs.

Previous steps & gathered info:
"""

from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
# Initialize HuggingFace embeddings with a good model for semantic search
embeddings_store = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'}, # Use GPU if available
    encode_kwargs={'normalize_embeddings': True} # Normalize embeddings for better similarity search
)
web_text_vectorstore: Any = Field(default=None)
try:
    web_text_vectorstore = FAISS.load_local("./gpt-oss-20b_rtx3090_tools/books/lorem_ipsum/vectorstore/vectorstore.db", embeddings_store, allow_dangerous_deserialization=True)
except Exception as e:
    print("Error: Loading vectorstore failed:", e)
# Connect query to FAISS index using a retriever
retriever = web_text_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)
agent_memory = VectorStoreRetrieverMemory(retriever=retriever) 

# Create json chat agent
agent = create_json_chat_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
    stop_sequence=["STOP"],
    template_tool_response="{observation} Remember to add the STOP word after each snippet",
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=agent_memory, max_iterations=50, handle_parsing_errors=True)
read_text_to_be_split = agent_executor.invoke({"input": "Write a complete performance optimized python program to calc prime numbers and prove it using the ExecutePythonCode Tool for the first 4711 numbers?"})

print("history " + "\n" + read_text_to_be_split['history'])
print("output " + "\n" + read_text_to_be_split['output'])

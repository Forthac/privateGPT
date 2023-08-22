from threading import Thread
import sys
from privateGPT import (
    RetrievalQA, HuggingFaceEmbeddings, Chroma,
    LlamaCpp, GPT4All, load_dotenv, StreamingStdOutCallbackHandler
)
import os
import time
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

global query_entry
def initialize_environment():
    global embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch, target_source_chunks

    # Loading environment variables
    load_dotenv()

    # Setup code from privateGPT.py
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
    from constants import CHROMA_SETTINGS

def initialize_embeddings_and_database():
    global embeddings, db, retriever

    # Initialize embeddings and database
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

def initialize_llm():
    global llm

    # Prepare the LLM
    print("Loading Language Model - please wait ...")
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

def initialize_query_chain():
    global qa

    # Prepare the query chain
    print("Initializing query chain ...")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

def initialize_gui():
    global notebook

    # Create the notebook widget to hold the tabs
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH)

    # Update the GUI with the query interface and tree view
    create_query_interface()
    create_tree_view(notebook)

# Function to initialize the model and other components
def initialize_model():
    # Redirecting standard output to capture status messages
    sys.stdout = TextRedirector(console_output)

    initialize_environment()
    initialize_embeddings_and_database()
    initialize_llm()
    initialize_query_chain()
    initialize_gui()

    # Restore standard output
    sys.stdout = sys.__stdout__


# Class to redirect standard output to the console_output widget
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)

    def flush(self):
        pass

def create_query_interface():
    global query_entry

    # Create the first tab (existing interface)
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Query Interface")

    # Scrolled text widget for console output in the first tab
    console_output = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, state=tk.DISABLED)
    console_output.grid(row=0, column=0, columnspan=2, sticky="nsew")

    # Entry widget for query input in the first tab
    query_entry = tk.Entry(tab1)
    query_entry.grid(row=1, column=0, sticky="nsew")

    # Button to submit the query in the first tab
    submit_button = tk.Button(tab1, text="Submit", command=on_submit)
    submit_button.grid(row=1, column=1, sticky="nsew")


# Function to handle query submission
def on_submit():
    query = query_entry.get()
    if query == "exit" or query.strip() == "":
        return

    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    # Display the result in the console_output widget
    console_output.config(state=tk.NORMAL)
    console_output.insert(tk.END, f"\n\n> Question:\n{query}\n> Answer (took {round(end - start, 2)} s.):\n{answer}")

    # Print the relevant sources used for the answer
    for document in docs:
        console_output.insert(tk.END, f"\n> {document.metadata['source']}:\n{document.page_content}")
    console_output.config(state=tk.DISABLED)

    # Clear the query entry field
    query_entry.delete(0, tk.END)


# GUI setup
root = tk.Tk()
root.title("privateGPT GUI")

# Scrolled text widget for console output (main output field)
console_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.NORMAL)
console_output.pack(expand=True, fill=tk.BOTH)

# Start the initialization process in a separate thread
initialization_thread = Thread(target=initialize_model)
initialization_thread.start()

root.mainloop()
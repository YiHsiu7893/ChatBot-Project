## Import necessary libraries.
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.llamacpp import LlamaCpp
import time


# Load .pdf document.
loader = PyPDFDirectoryLoader("./Knowledge_base")
documents = loader.load()
#print(documents)

# Split the whole document into several chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
#print(chunks[0])

# Embedding model.
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Convert documents into vector representations.
vectorstore = Chroma.from_documents(documents, embeddings)

# Retriever.
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# Instantiate the LlamaCpp language model.
llm = LlamaCpp(
    model_path= "D:/Downloads/BioMistral-7B.Q4_K_M.gguf",
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
    verbose=False,
    n_gpu_layers=33,
    n_ctx=900)


# Import necessary libraries.
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate


# Define prompt template.
template = """
<|context|>
I am a helpful health evaluator.
Based on the patient's statement, give a decimal between 0 and 1, where
0-0.2 means mild or none, 0.3-0.6 means moderate, and 0.7-1.0 means severe.
</s>
<|user|>
{query}
</s>
    <|assistant|>
"""

rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(template)
    | llm
    | StrOutputParser()
)


target_list = []
with open('./Knowledge_base/targets.txt', 'r') as file:
    for line in file:
        line = line.strip()
        target_list.append(line)


# A function to call for LLM
def llm_call(sent):
    queries = []
    scores = []

    for f in target_list:
        q = "Patient: "
        q += sent
        q += f"\nHow severe would you say my \"{f}\" is, on a scale of 0.0 to 1.0?"
        q += "Just give a number directly."

        queries.append(q)

    for i in range(len(queries)):
        # Generate response.
        response = rag_chain.invoke(queries[i])
        #print("\n", response)

        score = extract_score(response)   
        scores.append(score)

    return scores


# Extract numbers from LLaMA's response (text)
import re
def extract_score(string):
    number = re.findall(r'\d+\.\d+', string)
    if number:
        for i in number:
            return float(i)
    else:
        return 0.0

"""
Idea:
1. 從targets產生features (目前直接用現成的features)
2. 對每個feature，問"有這個症狀嗎?嚴重度?"，並retrieve from baseknowledge(目前是Symptom2Disease。可能需要另外自備?)
3. RAG問LLM，得到score array
4. Concatenate with other parts (in Models.py Process_Module)

start_time = time.time()
scores = llm_call("Standing or walking for long periods of time causes a lot of pain in my legs. I get cramps upon doing physical activities. There are bruise marks on my legs too.")
print("score:", scores)
end_time = time.time()

# 計算並打印運行時間
elapsed_time = end_time - start_time
print(f"Elapse time: {elapsed_time:.4f} s")
"""

## Import necessary libraries.
#from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.llamacpp import LlamaCpp

# Load .csv document.
loader = CSVLoader("./Symptom2Disease.csv", encoding="windows-1252")
documents = loader.load()


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
    verbose=False)


# Import necessary libraries.
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate


feature_list = []
with open('./features.txt', 'r') as file:
    for line in file:
        line = line.strip()
        feature_list.append(line)


# Define prompt template.
template = """
<|context|>
You are a helpful health scorer.\n
After hearing the patient's description below, give me a number between 0 and 1.\n
This is the rule of score: 0-0.2 is mild or none, 0.3-0.6 is moderate, and 0.7-1.0 is severe.\n\n
This is a patient's description:\n
---------------------\n
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


def llm_call(sent):
    scores = []
    queries = []

    for f in feature_list:
        q = sent
        q += "\n---------------------\n"
        q += f"If you are asked to give a score that represents the patient's severity of \"{f}\", how much will you give?"

        queries.append(q)


    for i in range(len(queries)):
        # Generate response.
        response = rag_chain.invoke(queries[i])
        #print(response)

        score = extract_score(response)
        
        scores.append(score)
        #scores.append(0)

    return scores


# Extract numbers from LLaMA's response(text)
import re
def extract_score(string):
    #number = re.findall(r'\d+\.\d+|\d+', string)
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
"""

#scores = llm_call("I have a rash on my legs that is causing a lot of discomforts. It seems there is a cramp and I can see prominent veins on the calf. Also, I have been feeling very tired and fatigued in the past couple of days.")
#print("score:", scores)

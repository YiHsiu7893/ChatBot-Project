## Import necessary libraries.
#from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp

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
Give a score that is only one number between 0 and 1.\n
The score number must be an decimals.\n
This is the rule of answer: 0-0.2 is mild or none, 0.3-0.6 is moderate, and above 0.7 is severe.\n
</s>
<|user|>
This is a patient‘s description:\n
---------------------\n
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


# Get a list of scores using RAG
def llm_call(sent):
    scores = []
    queries = []

    for f in feature_list:
        q = sent
        q += "\nGiven the information, you are a helpful health consultant.\n"
        q += f'Answer the question: Does the person described in the case have {f}? Do you think it is serious?'
        q += "Your answer must be a number. For example, 0.5."

        queries.append(q)


    for i in range(len(queries)):
        """
        # Generate response.
        response = rag_chain.invoke(queries[i])
        print(response)
        """

        #scores.append(response)
        scores.append(0)

    return scores


"""
Idea:
1. 從targets產生features (目前直接用現成的features)
2. 對每個feature，問"有這個症狀嗎?嚴重度?"，並retrieve from baseknowledge(目前是Symptom2Disease。可能需要另外自備?)
3. RAG問LLM，得到score array
4. Concatenate with other parts (done in Models.py Process_Module)
"""

from gradio_client import Client

target_list = []
with open('./Knowledge_base/targets.txt', 'r') as file:
    for line in file:
        line = line.strip()
        target_list.append(line)


client = Client("Walmart-the-bag/Phi-3-Medium", verbose=False)


# Extract numbers from LLaMA's response (text)
import re
def extract_score(string):
    scores = []
    number = re.findall(r': \d+\.\d+', string)

    for i in number:
        scores.append(float(i.split(": ")[-1]))

    return scores
    

def llm_call(sent):
    query = "Patient: "
    query += sent
    query += f"\nScore each disease on how they relate to the patient's symptom: {target_list}"
    query += "\nScores are decimal numbers between 0~1."

    result = client.predict(
		message=query,
		api_name="/chat"
    )

    print(result)
    return extract_score(result)


scores = llm_call("I have a rash on my legs that is causing a lot of discomforts. It seems there is a cramp and I can see prominent veins on the calf. Also, I have been feeling very tired and fatigued in the past couple of days.")
print("score:", scores)


"""
## another model
client = Client("huggingface-projects/llama-2-7b-chat")
result = client.predict(
		message=query,
		request="Scores are decimal numbers between 0~1",
		param_7=1,
		api_name="/chat"
)
"""

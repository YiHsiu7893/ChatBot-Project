# Import Necessary Libraries
import torch

from PreProcessor import normal_preprocess
from Model import BiLSTM
from Attention import attention_block
from Linguistic_Extract import gpt_call



### Input ###
text = input("What symptoms are you experiencing?\n")
#text = "I have a rash on my legs that is causing a lot of discomforts. It seems there is a cramp and I can see prominent veins on the calf. Also, I have been feeling very tired and fatigued in the past couple of days."



### Pre-processing Module ###
sent = normal_preprocess(text)

# Loaded vocabulary list
loaded_vocab = torch.load('Weights/vocab.pth')

# Convert the text to a sequence of word indices
sent_indices = [loaded_vocab[word] if word in loaded_vocab else loaded_vocab["UNK"] 
                for word in sent]

# Padding for same length sequence
if len(sent_indices)<31:
    sent_indices = sent_indices+[0]*(31-len(sent_indices))
    


### BiLSTM ###
embedding_dim = 256
hidden_dim = 128
num_layers = 2
class_num = 17

# Load pre-trained BiLSTM model
model = BiLSTM(len(loaded_vocab), embedding_dim, hidden_dim, num_layers, class_num)
model.load_state_dict(torch.load('Weights/model.pth'))



### Symptoms Feature Extraction Module ###



### Attention ###
attention1 = attention_block(200)         # attention1: for path1 use (GPT output)
attention2 = attention_block(hidden_dim)  # attention2: for path2 use



### Main ###
# Testing I: Path 1
embedded_out = gpt_call(text)
#att_out = attention1(embedded_out)
att_out = attention1(embedded_out.unsqueeze(0))
print("\npath 1 result:")
print(att_out)


# Testing II: Path 2
x = torch.tensor(sent_indices).unsqueeze(0)
probs = model(x)
_, predictions = probs.max(1)

idx2dis = torch.load('Weights/idx.pth')
print("\npath 2 result:")
print(idx2dis[predictions.item()])
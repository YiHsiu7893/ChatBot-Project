# Import Necessary Libraries
import torch

from Tokenizers import general_tokenizer
from Model import Path2_Module
from Attention import attention_block
from Linguistic_Ext import gpt_call
from otc_recmd import otc_recmd


### Input ###
#text = input("What symptoms are you experiencing?\n")
text = "I have a rash on my legs that is causing a lot of discomforts. It seems there is a cramp and I can see prominent veins on the calf. Also, I have been feeling very tired and fatigued in the past couple of days."



### Pre-processing Module ###
sent = general_tokenizer(text)

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
hidden_dim = 100
num_layers = 2
class_num = 17

# Load pre-trained BiLSTM model
model = Path2_Module(len(loaded_vocab), embedding_dim, hidden_dim, num_layers, class_num, 31) 
model.load_state_dict(torch.load('Weights/model.pth'))


"""
### Feature Extraction Module ###
# convert into tensor
extract_vec = torch.from_numpy(feat_extr(text, 'None', with_id = True, tokens = 7))
# print(len(extract_vec))
padding = torch.zeros((len(extract_vec), 256 - 201))
extract_vec_pad = torch.cat((extract_vec, padding), dim=1)
"""



### Attention ###
attention1 = attention_block(200)         # attention1: for path1 use (GPT output)



### Main ###
# Testing I: Path 1
embedded_out = gpt_call(text)
# att_out = attention1(embedded_out)
att_out = attention1(embedded_out.double().unsqueeze(0))
print("\npath 1 result:")
print(att_out)


# Testing II: Path 2
x = torch.tensor(sent_indices).unsqueeze(0)
text = [text]
probs = model.run(x, text, None, 'test')
_, predictions = probs.max(1)

idx2dis = torch.load('Weights/idx.pth')
print("\npath 2 result:")
print(idx2dis[predictions.item()])

otc = input("Do you want to see the OTC recommendations? (press y/Y to get recommendations, otherwise press any key)")
if otc == 'y' or otc == 'Y':
    otc_recmd(text)
print("Thank you for using our service!")

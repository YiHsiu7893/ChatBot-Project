# Import Necessary Libraries
import torch

from Tokenizers import tokenizer
from Models import Process_Module
from Recommender import otc_recmd


def main(text):
    ### Process Module ###
    # Hyperparameters
    embedding_dim = 256
    hidden_dim = 100
    num_layers = 2
    class_num = 17


    # Tokenize input sentence
    sent = tokenizer(text)

    # Load vocabulary list
    loaded_vocab = torch.load('Weights/vocab.pth')

    # Convert the text to a sequence of word indices
    sent_indices = [loaded_vocab[word] if word in loaded_vocab else loaded_vocab["UNK"] 
                    for word in sent]

    # Do padding for same length sequence
    if len(sent_indices)<31:
        sent_indices = sent_indices+[0]*(31-len(sent_indices))


    # Load pre-trained module
    model = Process_Module(len(loaded_vocab), embedding_dim, hidden_dim, num_layers, class_num, 31) 
    model.load_state_dict(torch.load('Weights/model.pth'))



    ### Main ###
    # Make a prediction
    x = torch.tensor(sent_indices).unsqueeze(0)
    text_list = [text]
    probs = model.run(x, text_list, None, 'test')
    _, predictions = probs.max(1)

    idx2dis = torch.load('Weights/idx.pth')
    print(idx2dis[predictions.item()])
    return idx2dis[predictions.item()]


if __name__ == '__main__':
    ### Input ###
    #text = input("What symptoms are you experiencing?\n")
    text = "I have a rash on my legs that is causing a lot of discomforts. It seems there is a cramp and I can see prominent veins on the calf. Also, I have been feeling very tired and fatigued in the past couple of days." 

    main(text)

    # Medicine recommendation
    otc = input("Do you want to see the OTC recommendations? (press y/Y to get recommendations, otherwise press any key)")
    if otc == 'y' or otc == 'Y':
        print(otc_recmd(text))

    # End of the Chatbot
    print("Thank you for using our service!")

# Basic utilities needed in the code

import torch
import spacy
from torchtext.data.metrics import bleu_score

def translate_sentence(model, sentence, german, english, device, max_length=50):
    """
    This function translates the input german sentence to the english sentence.
    German sentence --> German Vector --> Encoder --> context vector --> Decoder --> English Vector --> English Sentence

    :param model: the sequence-to-sequnce model
    :param sentence: the input "german" sentence
    :param german: the german Field object
    :param english : the english Field object
    :param device: cuda / cpu
    :param max_length : maximum length of the translated sentence
    """

    spacy_german = spacy.load("de_core_news_sm")

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_german(sentence)]

    else:
        tokens = [token.lower() for token in sentence]

    # insert the start and end sequence
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indicies = [german.vocab.stoi[token] for token in tokens]

    # (N, ) --> (1 X N)
    sentence_tensor = torch.LongTensor(text_to_indicies).unsqueeze(0).to(device)

    # Retrieve the hidden_state and cell_state from the encoder
    with torch.no_grad():
        hidden_state, cell_state = model.Encoder_LSTM(sentence_tensor)

    # start the decoding part using start sequence and the (hidden_state, cell_state)
    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden_state, cell_state = model.Decoder_LSTM(previous_word, hidden_state, cell_state)

            # shape received : 1 X 1 X |Eng_Vocab|; squeeze it
            # output = output.squeeze(0)

            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model stops predicting if it predicts <eos> token (index)
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    # We have the indicies of the translated sentence in english
    # Now, we will predict the sentence
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]

def bleu(data, model, german, english, device):
    """
    *** reference : https://www.youtube.com/watch?v=DejHQYAGb7Q ***
    :param data: the batch containing german and english sentences
    :param model: the model
    :param german: the german Field object
    :param english: the english Field object
    :param device: cuda / cpu
    """

    targets = []
    outputs = []

    for example in data:
        ger_sent = vars(example)["ger_sent"]
        eng_sent = vars(example)["eng_sent"]
        
        prediction = translate_sentence(model, ger_sent, german, english, device)

        # remove the <eos> token from the end
        prediction = prediction[:-1]

        targets.append([eng_sent])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

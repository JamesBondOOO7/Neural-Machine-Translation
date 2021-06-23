import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import utils
import spacy
import model


def tokenize_german(text):
    """
    tokenizer for German language
    """
    return [token.text for token in spacy_german.tokenizer(text)]

def tokenize_english(text):
    """
    tokenizer for English language
    """
    return [token.text for token in spacy_english.tokenizer(text)]

if __name__ == '__main__':

    
    # tokenizers for German and English
    spacy_german = spacy.load("de")
    spacy_english = spacy.load("en")

    # Lets build the vocab
    german = Field(tokenize=tokenize_german,
                    lower=True,
                    init_token="<sos>",
                    eos_token="<eos>"
    )

    # Field Object for English
    english = Field(tokenize=tokenize_english,
                    lower=True,
                    init_token="<sos>",
                    eos_token="<eos>"
    )

    # dataset object
    dataset = TabularDataset(path=r"C:\Users\manan\PycharmProjects\Advanced Deep Learning\Encoder Decoder\input\dataset.csv",
                            format='csv',
                            skip_header=True,
                            fields=[('ger_sent', german), ('eng_sent', english)]
    )

    # BUILDING THE VOCAB
    german.build_vocab(dataset, max_size=10000, min_freq=3)
    english.build_vocab(dataset, max_size=10000, min_freq=3)

    GERMAN_VOCAB = german.vocab
    ENGLISH_VOCAB = english.vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    # checkpoint = torch.load(r"C:\Users\manan\PycharmProjects\Advanced Deep Learning\Encoder Decoder\checkpoint\checkpoint-NMT-BEST.pth")
    # print(checkpoint)
    # my_model = checkpoint['model']

    # Let's create the model
    # ENCODER : 
    input_size_encoder = len(GERMAN_VOCAB)  # vocab size
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = float(0.5)
    
    encoder_lstm = model.Encoder(input_size_encoder, encoder_embedding_size, 
                            hidden_size, num_layers, encoder_dropout).to(device)

    # DECODER : 
    input_size_decoder = len(ENGLISH_VOCAB)
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = float(0.5)
    output_size = len(ENGLISH_VOCAB)
    
    decoder_lstm = model.Decoder(input_size_decoder, decoder_embedding_size, hidden_size, 
                            num_layers, decoder_dropout, output_size).to(device)


    my_model = model.Seq2Seq(encoder_lstm, decoder_lstm).to(device)
    # load the state dict of the model
    sd = torch.load(r"C:\Users\manan\PycharmProjects\Advanced Deep Learning\Encoder Decoder\checkpoint\checkpoint-NMT-BEST-SD.pth")
    my_model.load_state_dict(sd)

    # dataset object
    dataset = TabularDataset(path=r"C:\Users\manan\PycharmProjects\Advanced Deep Learning\Encoder Decoder\input\dataset.csv",
                            format='csv',
                            skip_header=True,
                            fields=[('ger_sent', german), ('eng_sent', english)]
    )

    sentence = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"
    model.to(device)

    # Let's translate some sentences
    print(utils.translate_sentence(my_model, sentence, german, english, device))
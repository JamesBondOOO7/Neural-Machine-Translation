import torch
import torch.nn as nn
import torchtext
import spacy
from torchtext.data.metrics import bleu_score
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.optim as optim

import testIterators
import model
import config
import utils
import engine


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
    spacy_german = spacy.load("de_core_news_sm")
    spacy_english = spacy.load("en_core_web_sm")

    # Field Object for German
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

    # 80% training
    train_dataset, test_dataset = dataset.split(split_ratio=0.80)

    # BUILDING THE VOCAB
    german.build_vocab(train_dataset, max_size=10000, min_freq=3)
    english.build_vocab(train_dataset, max_size=10000, min_freq=3)

    GERMAN_VOCAB = german.vocab
    ENGLISH_VOCAB = english.vocab

    print(f"German Vocab Size : {len(GERMAN_VOCAB)}")
    print(f"English Vocab Size : {len(ENGLISH_VOCAB)}")

    # set up the device to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_BATCH_SIZE = config.train_batch_size
    TEST_BATCH_SIZE = config.test_batch_size

    # Iterators
    train_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, test_dataset),
        batch_sizes=(TRAIN_BATCH_SIZE,TEST_BATCH_SIZE),
        sort_within_batch = True,
        sort_key=lambda x: len(x.ger_sent),
        device=device
    )

    # if we wanna explore the data in train and test iterators
    # use this function
    # testIterators.testing_Iterators(train_iterator, test_iterator, GERMAN_VOCAB, ENGLISH_VOCAB)

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

    # Let's train the model
    print("Model Training started :)")

    EPOCHS = config.epochs
    learning_rate = config.learning_rate

    epoch_loss = 0.0
    best_loss = 10**7
    best_epoch = -1
    optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)
    pad_idx = ENGLISH_VOCAB.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    early_stopping_counter = 0
    print(my_model, end="\n")

    # for checking the model at every step
    sample_sentence = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"

    train_losses = []
    test_bleu_scores = []
    print(utils.translate_sentence(my_model, sample_sentence, german, english, device))
    # for epoch in range(1):
        
    #     epoch_loss = engine.train(train_iterator, my_model, optimizer, criterion, len(ENGLISH_VOCAB), device)

    #     # Append the training loss
    #     train_losses.append(epoch_loss)
    #     print(f"Epoch : {epoch} ; Epoch Loss : {epoch_loss}")

    #     # print the bleu bleu score for testing # update to 1:100
    #     print(f"Testing Bleu Score : {utils.bleu_score(test_dataset[1:10], my_model, german, english, device)}")

    #     if epoch_loss < best_loss:
    #         best_loss = epoch_loss
    #         best_epoch = epoch

    #     else:
    #         early_stopping_counter += 1
        
    #     if early_stopping_counter > 5:
    #         print("Early Stopping...")
    #         break
    #     print(" :) ")
    #     break
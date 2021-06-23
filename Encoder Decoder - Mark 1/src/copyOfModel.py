import torch
import torch.nn as nn
import random

# ---------------------------- ENCODER ----------------------------
class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, drop_prob):
        """
        :param input_size: the size of the input sequence
        :param embedding_size: the embedding dimension
        :param hidden_size: the hidden dimension used in the LSTM model
        :param num_layers: number of layers in the LSTM model
        :param drop_prob: the probability of dropout
        """

        # self.param_dict = {
        #     'input_size' : input_size,
        #     'embedding_size' : embedding_size,
        #     'hidden_size' : hidden_size,
        #     'num_layers' : num_layers,
        #     'drop_prob' : drop_prob
        # }

        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(drop_prob)  # for Regularization

        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # the rnn cell
        self.rnn = nn.LSTM(input_size = embedding_size,
                        hidden_size = hidden_size,
                        num_layers = num_layers,
                        dropout=drop_prob,
                        batch_first=True
        )

    def forward(self, x):
        """
        :param x: the vector form of the sentence 
                  (containing the indicies mapped in the vocab)
        """

        # pass the data
        # N X T --> N X T X D
        x = self.dropout(self.embedding(x))

        output, (hidden_state, cell_state) = self.rnn(x)

        # return the context vectors
        # their shape : L X N X H (num_layers X batch_size X hidden_size)
        return hidden_state, cell_state




# ---------------------------- DECODER ----------------------------
class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, drop_prob, output_size):
        """
        :param input_size: the size of the input sequence
        :param embedding_size: the embedding dimension
        :param hidden_size: the hidden dimension used in the LSTM model
        :param num_layers: number of layers in the LSTM model
        :param drop_prob: the probability of dropout
        :param output_size: the output size of the linear layer after the decoding
        """

        # self.param_dict = {
        #     'input_size' : input_size,
        #     'embedding_size' : embedding_size,
        #     'hidden_size' : hidden_size,
        #     'num_layers' : num_layers,
        #     'drop_prob' : drop_prob,
        #     'output_size' : output_size
        # }

        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(drop_prob)  # for Regularization

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=drop_prob,
                            # batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):

        # unsqueeze x
        # shape becomes : 1 X N
        x = x.unsqueeze(0)

        # print("X in decoder . shape : ",x.shape)
        # 1 X N --> 1 X N X D
        x = self.dropout(self.embedding(x))

        # print("X in decoder's Embedding . shape : ",x.shape)

        # shape of outputs : 1 X N X H (1 X batch_size X Hidden_size)
        # shape of hidden and cell states : L X N X H
        outputs, (hidden_state, cell_state) = self.rnn(x, (hidden_state, cell_state))

        # 1 X N X H --> 1 X N X output_size
        predictions = self.fc(outputs)

        # 1 X N X output_size --> N X output_size
        predictions = predictions.squeeze(0)
        # print(f"Predictions : {predictions.shape}")

        # print("Hidden Decoder : ", hidden_state.shape)
        return predictions, hidden_state, cell_state




# ---------------------------- SEQUENCE-TO-SEQUENCE ----------------------------
class Seq2Seq(nn.Module):

    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        """
        :param Encoder_LSTM: the encoder part for the Seq2Seq model
        :param Decoder_LSTM: the decoder part for the Seq2Seq model
        """

        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, eng_vocab_size, tfr=0.5):
        """
        :param source: padded sentences in German
                       shape : [(sentence length German + some padding), #Sentences]
        :param target: padded sentences in English
                       shape : [(sentence length English + some padding), #Sentences]
        :param eng_vocab_size : size of the english vocab
        :param tfr: teach force ratio
        """

        # # Convert it into Batch Size X Sequence Length
        # source = source.permute(1, 0)
        target = target.permute(1, 0)

        batch_size = source.shape[0]
        target_len = target.shape[0]

        # print(f"Batch Size : {batch_size}, Target Length : {target_len}")

        outputs = torch.zeros(target_len, batch_size, eng_vocab_size).to(torch.device("cuda"))
        # print("Output shape : ", outputs.shape)

        # retaining the context vector from the encoder
        hidden_state, cell_state = self.Encoder_LSTM(source)

        # print("Hidden State : ", hidden_state.shape)

        # shape of x = shape of the target
        x = target[0]
        # print(x)

        # print("target[0].shape : ", x.shape)

        # print("Target shape : ", target.shape)

        # count = 1
        for i in range(1, target_len):

            # output : batch_size X |Eng_Vocab_Size|
            output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
            # print("Hidden State (Training) : ", hidden_state.shape)
            # print(output.shape)
            # print(outputs[i].shape)
            outputs[i] = output
            # best_guess = output.
            best_guess = output.argmax(1)  # the most suitable word embedding

            # print("Best guess.shape : ", best_guess.shape)
            # Teach force ratio
            # Either pass the next correct word from the dataset
            # or use the predicted word
            # count += 1
            # print(f"~~~~~~~~~~~~~~~~~~ {count} ~~~~~~~~~~~~~~~~~~")
            x = target[i] if random.random() < tfr else best_guess

        print(outputs.shape)
        # target_len X batch_size X english_vocab_size
        return outputs




if __name__ == '__main__':

    # ..................... Some testing code .....................

    # for encoder
    input_size_encoder = 5000  # vocab size
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = float(0.5)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_lstm = Encoder(input_size_encoder, encoder_embedding_size, 
                            hidden_size, num_layers, encoder_dropout).to(device)

    # print(encoder_lstm)

    # for decoder
    input_size_decoder = 4500
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = float(0.5)
    output_size = 4500
    
    decoder_lstm = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, 
                            num_layers, decoder_dropout, output_size).to(device)

    # print(decoder_lstm)

    model = Seq2Seq(encoder_lstm, decoder_lstm)
    print(model)
import torch
import torch.nn as nn
import config

def train(data_loader, model, optimizer, criterion, english_vocab_size, device):
    """
    This is the main training function that trains the model and
    returns training loss

    :param data_loader: this is the torch data loader
    :param model: model (encoder - decoder model)
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param criterion: loss function
    :param english_vocab_size: size of the english vocabulary
    :param device: this can be "cuda" or "cpu"
    """

    # set the model to training mode
    model.train()

    batch_loss = 0.0
    batches = 0
    for data in data_loader:

        input = data.ger_sent.to(device)
        target = data.eng_sent.to(device)

        input = input.permute(1, 0)
        
        # print()
        # print("``````````````````````````````````````````")
        # print(input.shape)
        # print(target.shape)
        # print('__________________________________________')
        # clear the gradients
        optimizer.zero_grad()

        # pass the input and target for model's forward method
        output = model(input, target, english_vocab_size)

        print("|||||||||||||||||||||||||||||||")
        output = output.permute(1, 0, 2)

        # print(output.shape)

        output = output[1:].reshape(-1, output.shape[2])

        target = target.permute(1, 0)
        target = target[1:].reshape(-1)

        # calculate the loss
        loss = criterion(output, target)

        # back-prop
        loss.backward()

        # clip the gradient value if it exceeds 1 => called NORM clipping  (https://www.youtube.com/watch?v=_-CZr06R5CQ)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # update the weight values
        optimizer.step()

        batches += 1.0
        batch_loss += loss.item()

    return batch_loss/batches


def evaluate(data_loader, model, criterion, device):
    """
    This function is used for returning loss

    :param data_loader: this is the torch data loader
    :param model: model (encoder - decoder model)
    :param criterion: loss function
    :param device: this can be "cuda" or "cpu"
    """

    batch_loss = 0.0
    batches = 0

    # put the model in evaluation mode
    model.eval()

    with torch.no_grad():

        for data in data_loader:

            input = data.ger_sent.to(device)
            target = data.eng_sent.to(device)

            # pass the input and target for model's forward method
            output = model(input, target, config.eng_vocab_size)

            loss = criterion(output, target)

            batches += 1.0
            batch_loss += loss.item()

    return batch_loss/batches

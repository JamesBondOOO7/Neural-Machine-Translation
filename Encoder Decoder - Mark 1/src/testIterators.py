def testing_Iterators(train_iterator, test_iterator, GERMAN_VOCAB, ENGLISH_VOCAB):
    """
    This function just prints the batches

    :param train_iterator: iterator for training
    :param test_iterator: iterator for testing
    :param GERMAN_VOCAB: the German vocab
    :param ENGLISH_VOCAB: the English vocab
    
    """
    for data in train_iterator:
        # print(f"Length : {data.ger_sent.shape}")  # "German :", *data.ger_sent, 
        # print(f"Length : {data.eng_sent.shape}")  # "English :", *data.eng_sent, 
        
        print("-------------- GERMAN SENTENCES ------------")
        print()
        temp = data.ger_sent.permute(1, 0)
        for ele in temp:
            for num in ele:
                print(GERMAN_VOCAB.itos[num.item()], end=" ")

            print()

        print()

        print("-------------- ENGLISH SENTENCES ------------")
        print()
        temp = data.eng_sent.permute(1, 0)
        for ele in temp:
            for num in ele:
                print(ENGLISH_VOCAB.itos[num.item()], end=" ")

            print()

        print()
        break

    for data in test_iterator:
        # print(f"Length : {data.ger_sent.shape}")  # "German :", *data.ger_sent, 
        # print(f"Length : {data.eng_sent.shape}")  # "English :", *data.eng_sent, 
        
        print("-------------- GERMAN SENTENCES ------------")
        print()
        temp = data.ger_sent.permute(1, 0)
        for ele in temp:
            for num in ele:
                print(GERMAN_VOCAB.itos[num.item()], end=" ")

            print()

        print()

        print("-------------- ENGLISH SENTENCES ------------")
        print()
        temp = data.eng_sent.permute(1, 0)
        for ele in temp:
            for num in ele:
                print(ENGLISH_VOCAB.itos[num.item()], end=" ")

            print()

        print()
        break

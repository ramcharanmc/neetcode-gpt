import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        positive_word_list = [sentence.split(" ") for sentence in positive]
        negative_word_list = [sentence.split(" ") for sentence in negative]
        combined = positive_word_list + negative_word_list
        
        vocab=sorted(list(set([word for word_list in combined for word in word_list])))
        
        #positive_id = torch.tensor([vocab.index(word)+1 for word in positive[0].split(" ")], dtype=torch.float)
        #negative_id = torch.tensor([vocab.index(word)+1 for word in negative[0].split(" ")], dtype=torch.float)
        
        encoded = []
        for words in combined:
            sent = []
            for word in words:
                sent.append(vocab.index(word)+1)
            encoded.append(torch.tensor(sent))

        return torch.nn.utils.rnn.pad_sequence(encoded, padding_value=0, batch_first=True)



        pass

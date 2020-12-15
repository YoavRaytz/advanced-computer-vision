import torch
import torch.nn as nn
from torch.nn.functional import embedding
from torchvision import models
import torchvision.models as models
from torch.functional import F  


class Encoder(nn.Module):
    def __init__(self, embed_size, train_backbone=False) -> None:
        super(Encoder, self).__init__()
        self.train_backbone = train_backbone
        self.model =  models.resnet152(pretrained=True)
        #Send the in feautres that's outputs from the CNN to RNN
        # replace the classifier with a fully connected embedding layer
        # self.model.classifier[1] = torch.nn.Linear(in_features=self.model.classifier[1].in_features, 
        #                                                                             out_features=embed_size) #for mobilenet
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size) #for resnets models
        self.freeze()
        self.relu = nn.ReLU() #after linear layer
        self.dropout = nn.Dropout(0.5) # reduce variance


    def forward(self, images):
        """
        The Encoder takes in images and basicly extracts the features with the backbone.
        Write the process in code below.
        """
        #Features extraction 
        output = self.model(images)
        
        return  self.dropout(self.relu(output))

    def freeze(self):
        for name, param in self.model.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                #Fine tuning one the last layer
                param.requires_grad = True
            else:
                param.requires_grad = self.train_backbone


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, features, captions):
        """
        Use the skeleton above and code the Decoder forward-pass
        **features are the extracted features we obtained from the bacbone.**
        Fill me!
        """
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_backbone=False) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)


    def forward(self, images, captions):
        features  = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption(self, image, vocabulary, max_length=50):

        """
        Our image caption inference!
        take in an image and vocabulary
        run the model on the image and output the caption!
        """
        result = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)

            states = None
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1) #Take the word with the hightest probability
                result.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [vocabulary.itos[idx] for idx in result]
        
            
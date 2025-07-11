from transformers import RobertaModel,RobertaConfig
import torch

# Defining downstream tasks
class RobertaClassification(torch.nn.Module):
    def __init__(self,config,num_labels):
        super().__init__()
        self.roberta = RobertaModel(config)

        # Define Classifier
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, encode_token):
        # Output from Roberta model
        output = self.roberta(encode_token)
        # Get Pooled output for [CLS] token
        pooled = output[1]
        # Feed the [CLS] output to the classifier layer
        logits = self.classifier(pooled)
        # Return classification results from Softmax
        return torch.softmax(logits, dim=1)




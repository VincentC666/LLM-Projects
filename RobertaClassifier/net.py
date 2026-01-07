from transformers import RobertaModel,RobertaConfig
import torch

# Defining downstream tasks
class RobertaClassification(torch.nn.Module):
    def __init__(self,config,num_labels):
        super().__init__()
        self.roberta = RobertaModel(config)

        # Define Classifier
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids,attention_masks):
        # Output from Roberta model
        with torch.no_grad():
            output = self.roberta(input_ids,attention_masks)
        out = self.classifier(output.last_hidden_state[:,0])
        #out = out.softmax(dim=1)
        return out
        # Get Pooled output for [CLS] token
        # pooled = output[1]
        # # Feed the [CLS] output to the classifier layer
        # logits = self.classifier(pooled)
        # return logits
        # Return classification results from Softmax
        # return torch.softmax(logits, dim=1)




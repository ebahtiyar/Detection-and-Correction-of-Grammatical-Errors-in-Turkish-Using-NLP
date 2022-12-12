from transformers import BertForTokenClassification
import torch
unique_labels = set()
unique_labels = {'QUE', 'COM', 'EMP', 'PER'}
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
label_all_tokens = True
class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('dbmdz/bert-base-turkish-128k-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


def return_model():
    model = BertModel()
    #change path after download models from drivers link
    model = torch.load(r'C:\Users\emreb\Documents\web_proje\models\model_punc\punctuation_model',map_location=torch.device('cpu'))
    return model
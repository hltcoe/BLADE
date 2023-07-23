import json
import math
import heapq
import torch
import operator

from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForMaskedLM

class Blade(torch.nn.Module):

    def __init__(self, model_type_or_dir):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]
        values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        return values

def process_text(texts, ids, model, tokenizer, device, reverse_voc, max_length, top_k):
  with torch.inference_mode():
    features = tokenizer(
        texts, return_tensors = "pt", max_length = max_length,
        padding = True, truncation = True
    )
    features = {key: val.to(device) for key, val in features.items()}
    doc_reps = model(**features)

  cols = [torch.nonzero(x).squeeze().cpu().tolist() for x in torch.unbind(doc_reps, dim = 0)]

  res = {}
  for col, doc_rep, id_ in zip(cols, doc_reps, ids):
    weights = doc_rep[col].cpu().tolist()

    if type(col) == list:
      weights_dict = {k : v for k, v in zip(col, weights)}
    else:
      weights_dict = {col : weights}

    tokids = heapq.nlargest(top_k, weights_dict, key = weights_dict.__getitem__)
    tokids = set(tokids)

    dict_blade = {
      reverse_voc[k]: round(v * 100)
      for k, v in weights_dict.items() if  k in tokids and round(v * 100) > 0
    }

    dict_blade = dict(sorted(dict_blade.items(), key = operator.itemgetter(1), reverse = True))

    if len(dict_blade.keys()) == 0:
      print("empty input =>", id_)
      dict_blade['"[unused993]"'] = 1

    res[id_] = dict_blade
  return res

if __name__=="__main__":
    parser = ArgumentParser(description='Generate bilingual vectors using BLADE model')
    parser.add_argument('--model_path', dest='model_path', required=True)
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=True)
    parser.add_argument('--is_query', dest='is_query', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Blade(args.model_path)
    
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    pids, texts = [], []
    with open(args.input) as f:
        for line in f:
            pid, text = line.strip().split("\t")
            pids.append(pid)
            texts.append(text.lower())

    reverse_voc = {v : k for k, v in tokenizer.vocab.items()}
    top_k = int(len(tokenizer) * 0.01) # Only preserving the top 10% of the highest weights
    max_length = 32 if args.is_query else 256

    with open(args.output, "w") as f:
        for i in tqdm(range(0, len(texts), batch_size), total = math.ceil(len(texts) / batch_size)):
            text_batch = texts[i:i+batch_size]
            pid_batch = ids[i:i+batch_size]
            res = process_text(text_batch, pid_batch, model, tokenizer, device, reverse_voc, max_length, top_k)
            
            for id_, text in zip(pid_batch, text_batch):
                dict_ = dict(id=id_, content=text, vector=res[id_])
                f.write(json.dumps(dict_, ensure_ascii=False)+"\n")

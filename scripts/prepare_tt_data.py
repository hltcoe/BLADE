import json
import pickle
import argparse
from tqdm import tqdm

def get_map_dict(path):
    map_dict = {}
    with open(path) as f:
        for line in tqdm(f):
            id_, text = line.strip().split("\t")
            text = text.strip()
            text = text.lower()
            map_dict[id_] = text
    return map_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create translate-train file for tevatron.')
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)
    parser.add_argument('--passage_file', dest='passage_file', required=True)

    args = parser.parse_args()
    passage_dict = get_map_dict(args.passage_file)
    
    query_data = {}
    with open(args.input) as f, open(args.output, "w") as g:
        for line in tqdm(f, total=398792):
            temp = json.loads(line)
            temp["positives"] = [passage_dict[key] for key in temp["positives_id"]]
            temp["negatives"] = [passage_dict[key] for key in temp["negatives_id"]]
            g.write(json.dumps(temp)+"\n") 


import os
import re
import json
import argparse
from tqdm import tqdm
from typing import Dict
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that generates document ranking using a ranked list of passages using MaxP approach.")
    parser.add_argument("--mapping", type=str, required=True)
    parser.add_argument("--rank_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cutoff", type=str, default=1000) 
    
    args = parser.parse_args()

    mapping_dict = {}
    with open(args.mapping) as f:
        for line in f:
           map_id, pass_id = line.strip().split("\t")
           mapping_dict[map_id] = pass_id 
    
    rank_dict = defaultdict(list)
    with open(args.rank_file) as f:
        for line in f:
            lsplit = line.strip().split(" ")
            qid = lsplit[0]
            map_id = lsplit[2]
            rank_dict[qid].append(mapping_dict[map_id])

    agg_dict = defaultdict(list)
    seen_pairs = set()
    
    for qid, pass_list in rank_dict.items():
        rank = 1
        for pass_id in pass_list:
            doc_id = pass_id.split("_")[0]
            
            if (qid, doc_id) not in seen_pairs:
                agg_dict[qid].append(doc_id)
                seen_pairs.add((qid, doc_id))
                
                if rank == args.cutoff: 
                    break
                
                rank += 1
                

    with open(args.output, "w") as f:
        for qid, doc_list in agg_dict.items():
            for rank, doc_id in enumerate(doc_list, start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{1001-rank}\tBLADE\n")


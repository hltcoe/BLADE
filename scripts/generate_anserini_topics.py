import json
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

def process(input_filename, output_filename):
    with open(input_filename) as input_file, open(output_filename, "w+") as output_file:
        for inp_line in tqdm(input_file):
            tmp = json.loads(inp_line)
            qid = tmp["id"]
            vector = " ".join([" ".join([key]*val) for key, val in tmp["vector"].items()])
            line = f"{qid}\t{vector}\n"
            output_file.write(line)

def main():
    parser = ArgumentParser(description='Convert BLADE query vectors into an Anserini compatible query file')
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)
    args = parser.parse_args()

    process(args.input, args.output)

if __name__ == "__main__":
    main()


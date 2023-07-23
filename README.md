# Running BLADE model

We provide the instructions to run a fine-tuned BLADE model on a CLIR collection.  The BLADE-C checkpoints are available on huggingface for English queries and Chinese, Persian, and Russian document languages (more document languages to follow!) which can be found here:

-   Chinese: https://huggingface.co/srnair/blade-en-zh
-   Persian: https://huggingface.co/srnair/blade-en-fa
-   Russian: https://huggingface.co/srnair/blade-en-ru


## Data Prep

For each experiment, we use `$EXP_DIR` as the root directory where the data files will be written. The language of the collection can be set using the`$LANG` variable which can take values as zh (Chinese), fas (Farsi) or rus (Russian).

We need to split the raw text documents into passages to apply the BLADE ranking model. The code to create passages is shown below

```
python scripts/create_passage_corpus.py \
 --root $EXP_DIR \
 --corpus path_to_collection.jsonl \
 --length 256 \
 --stride 128
```

This script assumes the CLIR document collection has been preprocessed into a JSONL file in the format shown below:

```
{
    "id": document_identifier,
    "title": title of the document content (if any),
    "text": actualy body of the document
}

```

The script uses the JSONL corpus file to creates two files `collection_passages.tsv`and `mapping.tsv` in the root directory.

By default, the passages are set to 256 tokens long with a stride of 128 tokens to create overlapping sequences. The length and the stride can be changed by setting `--length` and `--stride` respectively. We recommend setting length not more than 256 since BLADE model was trained on a  maximum sequence length of 256.

In case the collection already exists in JSONL format, the script also supports setting custom document identifier `--docid` , title field `--title` and text field `--text` from the JSONL file.

## CLIR indexing

The process of indexing a CLIR collection with a fine-tuned BLADE model is a two-step process.

1. We generate the bilingual sparse weights for the preprocessed passages using the fine-tuned BLADE model.

2. We store the generated weights into a sparse index using Anserini.

### Step 1: Generate BLADE vectors for a collection

We first create bilingual sparse vectors for a CLIR document collection using the code as shown below

```
mkdir -p $EXP_DIR/doc_outputs

python inference.py \
    --model_path=$MODEL_PATH \
    --input=$EXP_DIR/path_to_collection.tsv \
    --output=$EXP_DIR/doc_outputs/blade.jsonl \
    --batch_size=128
```

The input here is the `collection.tsv` file generated as part of the data prep process above.
The script outputs the blade vectors in JSONL file which is in a compatible format to be indexed by Anserini.

### Step 2: Index BLADE vectors using Anserini

We create a sparse index using Anserini as shown below

```
mkdir -p $EXP_DIR/indexes

sh path_to_anserini_target/appassembler/bin/IndexCollection \
    -collection JsonVectorCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 16 \
    -input collection \
    -index $EXP_DIR/indexes/blade \
    -impact \
    -pretokenized
```

You can tweak the `threads` parameter depending on the machine for faster execution time.


## CLIR Retrieval

Once indexing finishes, we generate a ranked list for the raw queries using the BLADE model. 
Similar to indexing, retrieval is a two-step process

1.  We generate the bilingual sparse weights for a query using the fine-tuned BLADE model.
2.  We perform retrieval using the generated query weights with Anserini.

### Step 1: Generate BLADE vectors for queries

We first create bilingual sparse vectors for queries using the code as shown below

```
mkdir -p $EXP_DIR/runs

python inference.py \
    --model_path=$MODEL_PATH \
    --input=path_to_queries.tsv \
    --output=$EXP_DIR/runs/blade.jsonl \
    --batch_size=128 \
    --is_query
```

The script takes as input a tsv file in the format `{query_id}\t{query_text}` and outputs blade vectors in JSONL file format.


### Step 2: Retrieval using Anserini

We need to convert the BLADE vectors into a compatible Anserini format first using the code shown below:

```
python generate_anserini_queries.py \
    --input=$EXP_DIR/runs/blade.jsonl \
    --output=$EXP_DIR/runs/blade_anserini.tsv
```

To retrieve indexed passages using the generated queries, we can run Anserini as below:

```
sh path_to_anserini_target/appassembler/bin/SearchCollection \
    -index $EXP_DIR/indexes/blade \
    -topics $EXP_DIR/runs/blade_anserini.tsv \
    -topicreader TsvInt \
    -output $EXP_DIR/runs/blade_passage_ranking.trec \
    -impact \
    -pretokenized \
    -hits 10000
```

To generate a document ranking file from the passage ranking file, you will need to run this script:

```
python scripts/aggregate_passage_scores.py \
    --rank_file $EXP_DIR/runs/blade_passing_ranking.trec \
    --mapping $EXP_DIR/mapping.tsv \
    --output $EXP_DIR/runs/blade_doc_ranking.trec
```

Evaluation can be performed using  `trec_eval`:

```
trec_eval path_to_qrels $EXP_DIR/runs/blade_doc_ranking.trec -m map -m recall.100 -c
```

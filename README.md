# BLADE
BLADE (Bilingual Lexical AnD Expansion model) is a neural CLIR method, which creates sparse bilingual vectors for queries and documents in two different languages.

BLADE is powered by a bilingual language model, created by pruning a multilingual pretrained language model (mBERT).

# Installation
BLADE requires Python 3.9+ and PyTorch 1.9+ and uses [HuggingFace Transformers](https://github.com/huggingface/transformers) library.

```
conda env create -f environment_release.yml
conda activate blade
```

BLADE also requires [Anserini](https://github.com/castorini/anserini/tree/anserini-0.14.1) for indexing and retrieval.
Please refer to the link above for installing Anserini.

# Training BLADE model

BLADE model can be trained in a two-step process.

## Step 1: Intermediate Pretraining

We start with a pruned bilingual language model consisting of query and document language and perform an intermediate pretraining using aligned texts expressed in their native language. 

The pruned bilingual models are available for most of the CLIR documents collections used in the paper.
- French: https://huggingface.co/Geotrend/bert-base-en-fr-cased
- Italian: https://huggingface.co/Geotrend/bert-base-en-it-cased
- German: https://huggingface.co/Geotrend/bert-base-en-de-cased
- Spanish: https://huggingface.co/Geotrend/bert-base-en-es-cased
- Chinese: https://huggingface.co/Geotrend/bert-base-en-zh-cased
- Russian: https://huggingface.co/Geotrend/bert-base-en-ru-cased

For Persian, we create a pruned bilingual model using the steps detailed in the [Geotrend codebase](https://github.com/Geotrend-research/smaller-transformers) and release it on the HuggingFace [hub](https://huggingface.co/srnair/bert-base-en-fa-cased) 

Assuming we have aligned pairs of text (parallel sentences/passages or comparable passages) in the format `{query_language_text}\t{document_language_text}`, we preprocess the dataset using the code below:

```
python scripts/prepare_pretrain_data.py \
    --input path_to_aligned_text_in_tsv \
    --output path_to_pretraining_corpus
```

The intermediate pretraining can be performed using the following [tevatron] (https://github.com/texttron/tevatron) implementation.
```
python -m torch.distributed.launch --nproc_per_node=8 intermediate/src/tevatron/train.py \
    --model_name_or_path $MODEL_NAME \
    --train_dir path_to_pretraining_corpus \
    --output_dir path_to_intermediate_output_directory \
    --doc_lang $lang \
    --per_device_train_batch_size 24 \
    --train_n_passages 1 \
    --learning_rate 1e-5 \
    --max_steps 200000 \
    --save_steps 50000 \
    --dataloader_num_workers 2 \
    --do_train \
    --fp16 \
    --negatives_x_device \
    --freeze_layers
```

The `$MODEL_NAME` refers to the pruned bilingual model and `$lang` is a two-letter language code referring to the document language.


## Step 2: Retrieval Finetuning

Once we have the intermediate pretrained model, we adopt a translate-train approach to fine-tune the model for the downstream CLIR task.
We use MS MARCO collection specifically pairing the original English MS MARCO queries with the translated MS MARCO passages released as [mmarco](https://huggingface.co/datasets/unicamp-dl/mmarco/tree/main/data/google/collections).

We first prepare the translate-train dataset using the following code:

```
python prepare_tt_data.py \
    --input data/english_msmarco.jsonl \
    --passage_file path_to_mmarco_document_collection_tsv \
    --output path_to_finetuning_data
```

The script uses the English MS MARCO training data and creates a translate train dataset by replacing the positive and negative passages using the passage identifier mapping between the original and translated passage collection.

The retrieval finetuning can be performed using the following [tevatron] (https://github.com/texttron/tevatron) implementation.

```
python -m torch.distributed.launch --nproc_per_node=8 src/tevatron/driver/train.py \
    --model_name_or_path path_to_intermediate_pretrained_checkpoint \
    --train_dir $CORPUS \
    --output_dir path_to_finetuned_output_directory \
    --q_distil data/query_vector_distil.json \
    --p_distil data/passage_vector_distil.jsonl \
    --p_offset data/passage_vector_distil.offset \
    --per_device_train_batch_size 32 \
    --train_n_passages 2 \
    --learning_rate 1e-5 \
    --dataloader_num_workers 2 \
    --max_steps 100000 \
    --save_steps 50000 \
    --do_train \
    --fp16 \
    --negatives_x_device
```

# Running BLADE model

We provide the instructions to run a fine-tuned BLADE model on a CLIR collection.  The BLADE-C checkpoints are available on huggingface for English queries and Chinese, Persian, and Russian document languages (more document languages to follow!) which can be found here:

-   Chinese: https://huggingface.co/srnair/blade-en-zh
-   Persian: https://huggingface.co/srnair/blade-en-fa
-   Russian: https://huggingface.co/srnair/blade-en-ru


## Data Prep

For each experiment, we use `$EXP_DIR` as the root directory where the data files will be written. The language of the collection can be set using the`$LANG` variable which can take values as zh (Chinese), fas (Farsi) or rus (Russian).

We need to split the raw text documents into passages to apply the BLADE ranking model. The code to create passages is shown below

```
mkdir -p $EXP_DIR

python scripts/create_passage_corpus.py \
 --model_name srnair/blade-en-${LANG}
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

The model path could be the fine-tuned model available on HuggingFace or a custom fine-tuned model path stored locally.
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

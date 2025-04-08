### Training Details

The data processing and model training steps are documented in `./data/process.ipynb`. Below is a detailed description of the workflow:

------

#### **Data Preparation**

1. **Data Format**:
   - Entities and relations are stored in TSV files:
     - `entities.tsv`: Maps entity IDs to their canonical names.
     - `relations.tsv`: Defines the relationship types used in the knowledge graph.
     - `train.tsv`, `valid.tsv`, `test.tsv`: Contain triples in `(head, relation, tail)` format for training, validation, and testing.
2. **Preprocessing**:
   - Entity/Relation Alignment: Raw IDs are mapped to internal indices compatible with DGL-KE.
   - Dataset Splitting: Triples are split into training (80%), validation (10%), and test sets (10%) using stratified sampling to maintain relation distribution.

------

#### **Model Training**

We utilize 

[**DGL-KE**,]: https://github.com/awslabs/dgl-ke

 a high-performance library for training knowledge graph embeddings. Below are the key steps:

1. **Configuration**:
   Modify training hyperparameters in the script or command line. Example configuration:

   bash

   复制

   ```
   dglke_train \
     --dataset test4 \
     --data_path ./data/ \
     --model_name TransE \
     --batch_size 1024 \
     --neg_sample_size 256 \
     --hidden_dim 512 \
     --gamma 12.0 \
     --lr 0.1 \
     --max_step 100000 \
     --log_interval 1000 \
     --eval_interval 5000 \
     --gpu 0
   ```

2. **Supported Models**:
   Choose from `TransE`, `RotatE`, `DistMult`, `ComplEx`, or `DESCAL`. Each model is optimized for specific relation patterns (e.g., symmetry, inversion).

3. **Execution**:
   Run the training script from the command line. The trained model will be saved in `./data/` 

------

#### **Evaluation**

1. **Metrics**:
   - **Mean Rank (MR)**: Average rank of true triples among corrupted candidates.
   - **Hits@N**: Percentage of true triples ranked in the top *N* (e.g., Hits@1, Hits@10).
2. **Validation/Testing**:
   DGL-KE automatically evaluates the model on the validation/test sets during training. 

------

#### **Deployment for Prediction**

1. **Model Export**:
   Trained models are serialized and deployed via a Flask API in `./predict/main.py`.
2. **API Endpoint**:
   The service exposes a `GET /linkPredict` endpoint with parameters:
   - `start`: Head entity ID (e.g., `P25054*P24941` for composite keys).
   - `end`: Tail entity ID.
   - `rel`: Relation type (must match an entry in `relations.tsv`).
   - `model`: Model name (e.g., `TransE`).

------

#### **Example Usage**

bash

复制

```
# Predict the likelihood of a relation between two entities using TransE
cd ./predict
python app.py --port 5000

curl "http://localhost:5000/linkPredict?start=P25054&end=P24941&rel=located_in&model=TransE"
```

------

#### **Notes**

- Ensure the model checkpoint matches the `model` parameter in the API request.
- For composite entity IDs, concatenate primary keys with `*` (e.g., `P25054*P24941`).
- Refer to `relations.tsv` for valid relation types before making requests.
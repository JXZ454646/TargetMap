from flask import Flask, request, jsonify
import os
import subprocess
from flask_cors import CORS  # Import Flask-CORS
import csv
import json

app = Flask(__name__)
CORS(app)  # Enable cross-origin support

def process_results(file_path):
    """
    Process the result.tsv file, excluding cases where head entity and tail entity are identical.
    """
    filtered_results = []
    with open('entity_id_dict.json', 'r') as entity_file:
        entity_dict = json.load(entity_file)
    

    with open(file_path, 'r') as result_file:
        reader = csv.reader(result_file, delimiter='\t')
        header = next(reader)  # Read header
        for row in reader:
            if row:
                head, rel, tail, score = row
                if head != tail:  # Exclude cases where head entity and tail entity are identical
                    filtered_results.append({
                        "head": entity_dict.get(head),
                        "rel": rel,
                        "tail": entity_dict.get(tail),
                        "score": float(score),
                        "head_ID": head,
                        "tail_ID": tail,
                    })
    return filtered_results

def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, check=True, text=True)
        memory_free = [int(x) for x in result.stdout.strip().split('\n')]
        free_gpu = memory_free.index(max(memory_free))
        return free_gpu
    except Exception as e:
        app.logger.error(f"Error getting free GPU: {e}")
        return 0 

@app.route('/linkPredict', methods=['GET'])
def process_string():
    start = request.args.get('start')
    end = request.args.get('end')
    rel = request.args.get('rel')
    model = request.args.get('model')
    
    chebi_list = start.split('*') if start else []
    end_list = end.split('*') if end else []
    rel_list = rel.split('*') if rel else []
    
    entities_in_file = set()
    with open('../data/test4/entities.tsv', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if row:  # Ensure row is not empty
                entities_in_file.add(row[1])
                
    chebi_list = [entity for entity in chebi_list if entity in entities_in_file]
    if not chebi_list:
        return jsonify({"error": "No chebi_string provided"}), 400
    if not os.path.exists('../data/test4'):
        os.makedirs('../data/test4')

    file_path = os.path.join('../data/test4', 'head.list')
    with open(file_path, 'w') as f:
        for item in chebi_list:
            f.write(f"{item}\n")
    
    end_list = [entity for entity in end_list if entity in entities_in_file]
    file2_path = os.path.join('../data/test4', 'tail.list')
    if len(end_list) > 0:
        with open(file2_path, 'w') as f:
            for item in end_list:
                f.write(f"{item}\n")

    if((not 'all' in rel_list ) and len(rel_list) > 0):
        rel_list = [item for item in rel_list if item != 'all']
        file3_path = os.path.join('../data/test4', 'rel.list')
        with open(file3_path, 'w') as f:
            for item in rel_list:
                f.write(f"{item}\n")
    else:
        rel_list = []
                
    model_paths = {
        "TranSE": "../TransE_l2",
        "RotatE": "../RotatE",
        "DistMult": "../DistMult",
        "DESCAL": "../RESCAL",
        "ComplEx": "../ComplEx",
    }
                
    model_path = model_paths.get(model)
    free_gpu = get_free_gpu()
    app.logger.debug(f"Using GPU: {free_gpu}")

    if(len(end_list) == 0 and len(rel_list) == 0):
        command = (
            f"DGLBACKEND=pytorch dglke_predict --model_path {model_path} --format 'h_*_*' --exec_mode 'batch_head' "
            f"--data_files ../data/test4/head.list --score_func logsigmoid --topK 5 --raw_data "
            f"--entity_mfile ../data/test4/entities.tsv --rel_mfile ../data/test4/relations.tsv --gpu {free_gpu}"
        )
    elif(len(end_list) >=0 and len(rel_list) == 0):
        command = (
            f"DGLBACKEND=pytorch dglke_predict --model_path {model_path} --format 'h_*_t' --exec_mode 'batch_head' "
            f"--data_files ../data/test4/head.list ../data/test4/tail.list --score_func logsigmoid --topK 5 --raw_data "
            f"--entity_mfile ../data/test4/entities.tsv --rel_mfile ../data/test4/relations.tsv --gpu {free_gpu}"
        )
    elif(len(end_list) == 0 and len(rel_list) > 0):
        command = (
            f"DGLBACKEND=pytorch dglke_predict --model_path {model_path} --format 'h_r_*' --exec_mode 'batch_head' "
            f"--data_files ../data/test4/head.list ../data/test4/rel.list --score_func logsigmoid --topK 5 --raw_data "
            f"--entity_mfile ../data/test4/entities.tsv --rel_mfile ../data/test4/relations.tsv --gpu {free_gpu}"
        )
    else:
        command = (
            f"DGLBACKEND=pytorch dglke_predict --model_path {model_path} --format 'h_r_t' --exec_mode 'batch_head' "
            f"--data_files ../data/test4/head.list ../data/test4/rel.list ../data/test4/tail.list --score_func logsigmoid --topK 5 --raw_data "
            f"--entity_mfile ../data/test4/entities.tsv --rel_mfile ../data/test4/relations.tsv --gpu {free_gpu}"
        )

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Command '{e.cmd}' returned non-zero exit status {e.returncode}. Output: {e.output}"}), 500
    result_content = process_results('result.tsv')
    return jsonify({"result": result_content}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
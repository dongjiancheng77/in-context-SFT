import os
import json
import sys
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel
import torch


@torch.no_grad()
def bert_embedding(texts, batch=100):
    tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')
    model = AutoModel.from_pretrained('../models/bert-base-uncased').cuda()
    encoded_texts = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=96)
    encoded_texts = encoded_texts.to("cuda")
    cls_hid_li = []
    i = 0
    while i < len(texts):
        last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                          attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
        cls_hids = last_hids[:,0,:].squeeze()
        cls_hid_li.append(cls_hids)
        i += batch
    cls_hids_tensor = torch.concat(cls_hid_li, dim=0)
    return np.array(cls_hids_tensor.cpu())

def torch_euclidean_dist(x, y):
    return torch.norm(x - y, dim=1)
def greedy_tsp(embeddings):
    if torch.cuda.is_available():
        embeddings = torch.tensor(embeddings).float().cuda()
    else:
        embeddings = torch.tensor(embeddings).float()

    n = len(embeddings)
    visited = [False] * n
    path = [0]
    visited[0] = True

    # dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # # 将距离矩阵转换为上三角矩阵，排除对角线（自身距离为0），并将其转换为一维数组
    # upper_tri_dist = dist_matrix.triu(diagonal=1).flatten()
    # # 移除零元素（自身到自身的距离为0，但在上三角矩阵中被包含）
    # all_distances = upper_tri_dist[upper_tri_dist > 0]
    # all_distances_cpu = all_distances.cpu().numpy()

    # threshold = np.percentile(all_distances_cpu, 2)
    threshold = 6.7
    y=0
    for x in range(1, n):
        print('x:',x)
        last = path[-1]
        last_embedding = embeddings[last].unsqueeze(0)
        distances = torch_euclidean_dist(last_embedding, embeddings).cpu().numpy()

        distances[last] = np.inf  # Avoid comparing to itself

        # Mark visited nodes to prevent reselection
        for i in range(n):
            if visited[i]:
                distances[i] = np.inf

        # Ensure the selected node distance is greater than the threshold
        next_index = np.argmin(distances)
        z=0

        max_attempts = 1000  # 设置一个最大尝试次数
        attempts = 0  # 初始化尝试次数计数器

        while True:
            valid = True
            # 检查当前的next_index是否小于阈值

            # 检查路径中最后1, 2, 3, 4个点
            for back in range(1, min(5, len(path)+1)):
                if torch.norm(embeddings[path[-back]] - embeddings[next_index]).item() < threshold:
                    valid = False
                    break

            if valid:
                print('false_attempts:', y)
                break
            else:
                y += 1
                distances[next_index] = np.inf
                next_index = np.argmin(distances)

            attempts += 1  # 更新尝试次数
            if attempts >= max_attempts:
                print(f"Exiting loop after {max_attempts} attempts.")
                break


        
        path.append(next_index)  # Add to path
        visited[next_index] = True
    print(threshold)
    return path

def split_input_file(input_file, parts=8):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    chunk_size = len(data) // parts
    for i in range(parts):
        with open(f'part_{i+1}.json', 'w') as file:
            json.dump(data[i*chunk_size:(i+1)*chunk_size] if i < parts - 1 else data[i*chunk_size:], file, indent=2, ensure_ascii=False)

def process_parts(parts=8):
    results = []
    for i in range(parts):
        with open(f'part_{i+1}.json', 'r') as file:
            data = json.load(file)
        text_list = [d["instruction"] for d in data]
        embeddings = bert_embedding(text_list)
        order = greedy_tsp(embeddings)
        ordered_data = [data[index] for index in order]
        results.extend(ordered_data)
    return results

def main(input_file, output_file):
    split_input_file(input_file)
    ordered_results = process_parts()
    with open(output_file, 'w') as file:
        json.dump(ordered_results, file, indent=2, ensure_ascii=False)
    print('Processing complete. Output saved to', output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py input_file output_file")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)

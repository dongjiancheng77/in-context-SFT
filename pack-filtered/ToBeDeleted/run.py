# 多个修改
#阈值
import os
import json
import sys
import numpy as np
from transformers import BertTokenizer, BertModel,AutoModel
import torch

@torch.no_grad()
def bert_embedding(texts,batch=100):

    tokenizer = BertTokenizer.from_pretrained('/home/nfs02/dongjc/MoDS/models/bert-base-uncased')
    model = AutoModel.from_pretrained('/home/nfs02/dongjc/MoDS/models/bert-base-uncased').cuda()
    encoded_texts = tokenizer(texts,return_tensors="pt",truncation=True,padding=True,max_length=96)
    encoded_texts =  encoded_texts.to("cuda")
    cls_hid_li = []
    i= 0
    x = 0
    while i < len(texts):
        last_hids = model(input_ids=encoded_texts["input_ids"][i:i+batch],
                          attention_mask=encoded_texts["attention_mask"][i:i+batch])['last_hidden_state']
        cls_hids = last_hids[:,0,:].squeeze()
        cls_hid_li.append(cls_hids)
        i+= batch
        print(i)
        x += batch
        if x > 5000:
            x-=5000
            print('x',x)
            torch.cuda.empty_cache()
    # 将所有文本的embedding连成特征矩阵
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

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # 将距离矩阵转换为上三角矩阵，排除对角线（自身距离为0），并将其转换为一维数组
    upper_tri_dist = dist_matrix.triu(diagonal=1).flatten()
    # 移除零元素（自身到自身的距离为0，但在上三角矩阵中被包含）
    all_distances = upper_tri_dist[upper_tri_dist > 0]
    all_distances_cpu = all_distances.cpu().numpy()

    threshold = np.percentile(all_distances_cpu, 2)
    print(threshold)
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

        # Adjust the loop to check distances from the last 4 points in the path if they exist
        while True:
            valid = True
            # Check current next_index against the threshold
            if distances[next_index] < threshold:
                valid = False
            
            # Check last 1, 2, 3, and 4 points in the path
            for back in range(1, min(5, len(path))):  # Checks path[-2], path[-3], path[-4] if they exist
                if len(path) > back and torch.norm(embeddings[path[-back]] - embeddings[next_index]).item() < threshold:
                    valid = False
                    break
            if valid:
                print('valid:',y)
                break
            else:
                # print('not-valid')
                y += 1
                distances[next_index] = np.inf
                next_index = np.argmin(distances)
        # if z>1000:
        #     break

        
        path.append(next_index)  # Add to path
        visited[next_index] = True
    print('y:',y)
    return path

def print_distances(embeddings, order):
    embeddings_tensor = torch.tensor(embeddings).float()
    if torch.cuda.is_available():
        embeddings_tensor = embeddings_tensor.cuda()
    
    total_distance = 0
    for i in range(1, len(order)):
        # 使用张量索引而不是numpy数组
        distance = torch.norm(embeddings_tensor[order[i - 1]] - embeddings_tensor[order[i]], dim=0).item()
        total_distance += distance
        # print(f"Distance between {order[i - 1]} and {order[i]}: {distance}")

    print(f"Total distance: {total_distance}")

def main(input_file, output_file):
    with open(input_file, "r") as fp:
        data = json.load(fp)

    instruction_list = [d["instruction"] for d in data]
    print('Processing instructions...')
    if os.path.exists("bert_embeddingsm-bert.npy"):
        text_embedding = np.load("bert_embeddingsm-bert.npy")
        print(1)
    else:
        text_embedding = bert_embedding(instruction_list)
        print(2)
        np.save("bert_embeddingsm-bert.npy",text_embedding)
    # 打印原始顺序的距离
    print("Distances in the original order:")
    original_order = list(range(len(instruction_list)))
    print_distances(text_embedding, original_order)
    res = greedy_tsp(text_embedding)

    print("\nDistances in the TSP order:")
    print_distances(text_embedding, res)

    data_li = [data[index] for index in res]
    with open(output_file, "w") as fp:
        json.dump(data_li, fp, indent=2, ensure_ascii=False)

    print('Processing complete. Output saved to', output_file)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: script.py input_file output_file")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)

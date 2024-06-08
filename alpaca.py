import json

# 定义输入和输出文件的路径
input_file_path = 'MathInstruct/MathInstruct.json'
output_file_path = 'MathInstruct/MathInstruct_alpaca.json'

# 读取原始JSON文件
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    data = json.load(input_file)

# 转换数据到新的格式
alpaca_formatted_data = []
for item in data:
    # 提取问题和答案
    instruction = item['instruction']
    output = item['output']
    # 构建新的结构，包括空的"input"
    alpaca_entry = {
        "instruction": instruction,
        "input": "",
        "output": output
     }
alpaca_formatted_data.append(alpaca_entry)

# 写入新的JSON文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(alpaca_formatted_data, output_file, ensure_ascii=False, indent=4)

print(f"Data has been converted to alpaca format and saved to {output_file_path}")

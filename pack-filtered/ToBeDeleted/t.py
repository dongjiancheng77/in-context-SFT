import json

def process_compressed_json(file_path, output_path):
    # 读取压缩在一行中的JSON数据
    with open(file_path, 'r', encoding='utf-8') as file:
        compressed_json = file.read()
    
    # 将压缩的JSON数据解析为Python对象
    json_data = json.loads(compressed_json)
    
    # 将Python对象转化为格式化的JSON字符串
    formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
    
    # 将格式化后的JSON字符串写入新的文件
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(formatted_json)
    
    print(f"Formatted JSON data has been written to {output_path}")

# 输入文件路径和输出文件路径
input_file_path = 'CaR_selected_result.json'
output_file_path = 'CaR_selected_result1.json'

process_compressed_json(input_file_path, output_file_path)

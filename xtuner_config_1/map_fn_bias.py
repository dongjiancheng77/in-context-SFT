# 假设将该函数存放在./map_fn.py文件中
SYSTEM_ALPACA = ("You are a helpful, pattern-following assistant. For each snippet of text, prediction whether the person desribed has a good credit risk. The answer should be exactly 'yes' or 'no'.\n")
def custom_map_fn(example):
    if example.get('label') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'system': SYSTEM_ALPACA,
                'input': f"{example['description']}",
                'output': example['label']
            }]
        }
# 假设将该函数存放在./map_fn.py文件中
SYSTEM_ALPACA = ('You are a professional programer. Please provide the '
           'corresponding code based on the description of Human.\n')
def custom_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'system': SYSTEM_ALPACA,
                'input': f"{example['instruction']}\n{example['input']}",
                'output': example['output']
            }]
        }
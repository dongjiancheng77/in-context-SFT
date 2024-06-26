# 假设将该函数存放在./map_fn.py文件中
SYSTEM_ALPACA = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.\n')
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
import json

def load_data(filepath):
    """ Load data from a JSON file. """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_data(data, filepath):
    """ Save data to a JSON file. """
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def sort_data_by_instruction_output_length(data):
    """ Sort data based on the combined length of 'instruction' and 'output'. """
    return sorted(data, key=lambda x: len(x['instruction']) + len(x['output']))

def main():
    input_filepath = '../data/gsm8k_alpaca.json'
    output_filepath = './gsmbert_Sorted_batching-n4-2.json'

    # Load data from JSON file
    data = load_data(input_filepath)

    # Sort data by the combined length of 'instruction' and 'output'
    sorted_data = sort_data_by_instruction_output_length(data)

    # Save the sorted data to a new JSON file
    save_data(sorted_data, output_filepath)
    print("Data sorted and saved to", output_filepath)

if __name__ == "__main__":
    main()

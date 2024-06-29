import lmdeploy
import json
import random

# Create a pipeline with the model
pipe = lmdeploy.pipeline('/home/nfs02/model/llama-3-8b-instruct')

# Load the data from a JSON file
with open("/home/nfs04/dongjc/in-context-SFT/data/quality-evaluation_gsm.json", "r", encoding='utf-8') as file:
    data = json.load(file)

# Define a function to generate equivalence prompts
def equivalence_prompt(text1, text2):
    prompt = "Here are two possible answers:\n"
    prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
    prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"
    prompt += "Response: "
    return prompt

# Build lists of instructions and inputs
i = 0
while i < len(data) - 1:
    text1 = data[i]['instruction']
    text2 = data[i+1]['instruction']
    prompt1 = equivalence_prompt(text1, text2)
    response1 = pipe([prompt1])[0]
    print(response1)
    prompt2 = equivalence_prompt(text2, text1)
    response2 = pipe([prompt2])[0]
    if response1 == "entailment" and response2 == "entailment":
        i += 1
    else:
        j = random.randint(i + 2, len(data) - 1)
        # Swap the element at position i+1 with the element at position j
        data[i+1], data[j] = data[j], data[i+1]

# Write the modified data back to a JSON file
with open("./llamanligsm.json", "w", encoding='utf-8') as outfile:
    json.dump(data, outfile, indent=4)

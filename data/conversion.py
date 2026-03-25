import json

input_file = r"C:\Users\Administrator\Desktop\TINYAYA-COHERE\TINYAYADATASETS\instruct-dts\output_dataset.jsonl"
output_file = r"C:\Users\Administrator\Desktop\TINYAYA-COHERE\TINYAYADATASETS\instruct-dts\sinhalaQnA.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line in f_in:
        data = json.loads(line.strip())

        alpaca_entry = {
            "instruction": data.get("Question", ""),
            "input": "",
            "output": data.get("TranslatedAnswer", "")
        }

        f_out.write(json.dumps(alpaca_entry, ensure_ascii=False) + "\n")

print("✅ Conversion complete!")
print("Saved to:", output_file)
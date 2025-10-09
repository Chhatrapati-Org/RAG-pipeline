import json
import zipfile
import os 
from tqdm import tqdm

results_file = r"C:\Users\22bcscs055\Documents\ps04-rag-v2\retrieval_results4.json"
output_dir = r"C:\Users\22bcscs055\Downloads\Exported_Files4"
os.makedirs(output_dir, exist_ok=True)



# Open and read the JSON file
data = []
with open(results_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
for i in tqdm(data, desc="Exporting files", unit="file"):
    output_file = rf"{output_dir}\{i['query_num']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        d = {}
        d['query'] = i['query']
        d['response'] = i['response']
        json.dump(d, f, ensure_ascii=False, indent=4)

# Create a ZIP file of all exported JSON files
with zipfile.ZipFile(rf"{output_dir}\Astraq Cyber Defence_PS4.zip", 'w') as zip_ref:
    for file in tqdm(os.listdir(output_dir), desc="Zipping files", unit="file"):
        if file.endswith('.json'):
            output_file = os.path.join(output_dir, file)
            zip_ref.write(output_file, arcname=file)  # arcname=file ensures only filename is stored
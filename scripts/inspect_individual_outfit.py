import json
from pprint import pprint

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="mvasil/polyvore-outfits",
    filename="nondisjoint/train.json",
    repo_type="dataset",
)

with open(path, encoding="utf-8") as file:
    outfits = json.load(file)

print(f"{len(outfits)} outfits")
pprint(outfits[0])

metadata_path = hf_hub_download(
    repo_id="mvasil/polyvore-outfits",
    filename="polyvore_item_metadata.json",
    repo_type="dataset",
)

with open(metadata_path, encoding="utf-8") as file:
    metadata = json.load(file)

print("\nMetadata de las primeras 5 prendas:")
for item in outfits[0]["items"][:5]:
    item_id = str(item["item_id"])
    pprint(metadata[item_id])
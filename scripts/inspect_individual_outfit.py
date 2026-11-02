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

first_item_id = str(outfits[0]["items"][0]["item_id"])

print("\nMetadata de la primera prenda:")
pprint(metadata[first_item_id])
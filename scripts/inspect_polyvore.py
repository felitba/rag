import sys
from pathlib import Path
from pprint import pprint

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from home_made_rag.config import CONFIG

def main():
    settings = CONFIG["dataset"]
    
    dataset = load_dataset(
        settings["huggingface_id"],
        settings["configuration"],
        split=settings["split"],
        streaming=True,
    )
    
    print("Features:")
    pprint(dataset.features)
    
    row = next(iter(dataset))
    
    print("\nFields:")
    pprint(list(row.keys()))
    
    print("\nFirst row:")
    pprint(row)
    
if __name__ == "__main__":
    main()
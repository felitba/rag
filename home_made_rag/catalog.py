import json

from pathlib import Path
from urllib.parse import quote


from home_made_rag.config import (
    ITEMS_FILE,
    OUTFITS_FILE,
    IMAGES_DIR,
    ROOT_DIR,
)

class Catalog:
    """Carga prendas y outfits preparados para el proyecto"""
    def __init__(self):
        self.items = {}
        
        with ITEMS_FILE.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"La linea {line_number} debe contener un objeto JSON")
                item_id = item.get("item_id")
                if not isinstance(item_id, str) or not item_id.strip():
                    raise ValueError(f"item_id invalido en la linea {line_number}")
                
                if item_id in self.items:
                    raise ValueError(f"item_id duplicado: {item_id}")
                
                self.items[item_id] = item

        with OUTFITS_FILE.open("r", encoding="utf-8") as file:
            self.outfits = json.load(file)
            
        # esperamos un diccionario: outfit_id -> lista de item_ids
        if not isinstance(self.outfits, dict):
            raise ValueError("outfits.json debe contener un diccionario")
        
        self.outfits_by_item = {
            item_id: [] for item_id in self.items
        }
        
        for outfit_id, item_ids in self.outfits.items():
            if not isinstance(outfit_id, str) or not outfit_id.strip():
                raise ValueError("Se encontro un outfit_id invalido")
            
            if not isinstance(item_ids, list) or not item_ids:
                raise ValueError(f"El outfit {outfit_id} debe contener una lista de prendas")
            
            for item_id in item_ids:
                if not isinstance(item_id, str) or item_id not in self.items:
                    raise ValueError(f"El outfit {outfit_id} referencia prenda inexistente o invalida: {item_id!r}")
                
                if outfit_id not in self.outfits_by_item[item_id]:
                    self.outfits_by_item[item_id].append(outfit_id)
    
    def image_path(self, item_id:str) -> Path:
        item = self.items[item_id]
        relative_path = item.get("image_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"image_path invalido para la prenda {item_id}")
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(f"La ruta de la prenda {item_id} debe ser relative: {path}")
        
        absolute_path = (ROOT_DIR / path).resolve()
        
        images_root = IMAGES_DIR.resolve()
        
        if not absolute_path.is_relative_to(images_root):
            raise ValueError(f"La imagen de {item_id} esta fuera de {images_root}")
        
        return absolute_path
    
    def image_url(self, item_id:str) -> str:
        path = self.image_path(item_id)
        relative_path = path.relative_to(IMAGES_DIR.resolve())
        encoded_path = quote(relative_path.as_posix(), safe="/")
        return f"/images/{encoded_path}"
    
    def validate_images(self) -> None:
        for item_id in self.items:
            path = self.image_path(item_id)
            if not path.is_file():
                raise FileNotFoundError(f"No existe la imagen de la prenda {item_id}: {path}")
            
    def outfit_items(self, outfit_id:str) -> list[dict]:
        item_ids = self.outfits[outfit_id]
        return [self.items[item_id] for item_id in item_ids]
    
    def document(self, item_id: str) -> str:
        item = self.items[item_id]
        parts = [
            (item.get("title") or "").strip(),
            (item.get("description") or "").strip(),
        ]
        text = ". ".join(part for part in parts if part)
        return text or item["category"]
    
    def metadata(self, item_id:str) -> dict:
        item = self.items[item_id]
        return {
            "category": item["category"],
            "image_path": item["image_path"],
            "outfit_ids_json": json.dumps(self.outfits_by_item[item_id])
        }
    
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from extractor import procesar_carpeta # rol 1
from procesador_texto import chunkers_final # rol 2

# 1. Ruta de salida

salida = os.path.join(
    'entrega',
    'base_vectorial',
    'encoder_distiluse-base-multilingual-cased-v1'
)
os.makedirs(salida, exist_ok=True)

# 2. Extracción y chunking
documentos = procesar_carpeta('datos', fenomeno=1)
dic_chunks = chunkers_final(documentos)
metadata = list(dic_chunks.values())

# 3. Extracción de textos para la vectorización

encoder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
textos = [chunk['texto'] for chunk in metadata]
embeddings = encoder.encode(textos, normalize_embeddings=True, convert_to_numpy=True) # Vectorización y normalización L_2
embeddings = embeddings.astype('float32') # Requerido para FAISS

# 4. Índice FAISS usando la similitud coseno (IndexFlatIP - pag 12)

dimension = embeddings.shape[1] # Extrae la dimension preestablecidad del encoder (512)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
ruta_faiss = os.path.join(salida, 'index.faiss')
faiss.write_index(index, ruta_faiss) # pag 5

# 5. Generar el metadata.jsonl

ruta_metadata = os.path.join(salida, 'metadata.jsonl')

with open(ruta_metadata, 'w', encoding='utf-8') as f: #'w'writing. codificación utf-8 (Diacríticos y carácteres especiales). WIth para cerrar el archivo
  for chunk in metadata:
    meta_obj = {
        'doc_id': chunk['doc_id'],
        'chunk_id': chunk['chunk_id'],
        'texto': chunk['text'],
        'fuente': chunk.get('fuente', 'desconocido'),
        'num_tokens': len(chunk['texto'].split()),
    }
    f.write(json.dumps(meta_obj, ensure_ascii=False) + '\n')

print(
    f'Base vectorial creada exitosamente en: {salida}\n'
    f'- {ruta_faiss}\n'
    f'- {ruta_metadata}'
)
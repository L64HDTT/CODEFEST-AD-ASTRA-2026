import os
import json
import faiss
import numpy as np
import networkx as nx
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ----------------- ----------------------------------------
# 0. Carga de las consultas desde un archivo JSON
# ---------------------------------------------------------

def cargar_consultas(ruta_archivo="consultas.json"):
    """Carga las 50 consultas desde un archivo JSON a un diccionario."""
    print(f"Cargando consultas desde {ruta_archivo}...")
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        consultas_texto = json.load(f)
    return consultas_texto

# ---------------------------------------------------------
# 1. Carga de la metadata desde un archivo JSONL
# ---------------------------------------------------------

def cargar_tu_metadata(ruta_archivo):
    """Carga el metadata.jsonl donde el índice de la línea es el ID de FAISS."""
    metadata = {}
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        for i, linea in enumerate(f):
            # FAISS asigna IDs internos secuenciales (0, 1, 2...) que coinciden con las líneas
            metadata[str(i)] = json.loads(linea)
    return metadata

# ---------------------------------------------------------
# 2. Carga del Grafo de Conocimiento
# ---------------------------------------------------------

def cargar_grafo(ruta_archivo="grafo/grafo.graphml"):
    """Carga el grafo exportado en la etapa anterior."""
    print(f"Cargando grafo desde {ruta_archivo}...")
    if os.path.exists(ruta_archivo):
        return nx.read_graphml(ruta_archivo)
    else:
        print("Aviso: No se encontró el grafo. Se usará solo FAISS.")
        return None

# ---------------------------------------------------------
# 3. CARGA DE LA BASE VECTORIAL
# ---------------------------------------------------------

def cargar_base_vectorial():
    ruta_faiss = os.path.join(
      'entrega',
      'base_vectorial',
      'encoder_distiluse-base-multilingual-cased-v1',
      'index.faiss'
  )
    ruta_meta = os.path.join(
      'entrega',
      'base_vectorial',
      'encoder_distiluse-base-multilingual-cased-v1',
      'metadata.jsonl'
  )
    print("Cargando índice FAISS...")       
    index = faiss.read_index(ruta_faiss)
    
    print("Cargando almacén de metadata...")
    metadata = cargar_tu_metadata(ruta_meta)
    
    return index, metadata

# ---------------------------------------------------------
# 4. PROCESAMIENTO DE LAS CONSULTAS (RRF)
# ---------------------------------------------------------

def codificar_texto(texto, model):
    """Convierte el texto de la consulta en un vector numérico normalizado."""
    return model.encode([texto], normalize_embeddings=True)

def obtener_candidatos_grafo(texto_consulta, ner_pipeline, grafo):
    """Extrae entidades de la consulta y busca chunks relacionados en el grafo."""
    if not grafo or not ner_pipeline:
        return []
    
    # 1. NER en la consulta (Extraer entidades)
    resultados_ner = ner_pipeline(texto_consulta)
    entidades_consulta = [ent['word'].strip().upper() for ent in resultados_ner if ent['score'] > 0.70]
    
    frecuencia_chunks = {}
    
    # 2. Buscar entidades y sus vecinos de primer orden
    for ent in entidades_consulta:
        if ent in grafo:
            # Recuperar vecinos de primer orden (nodos conectados)
            vecinos = list(grafo.successors(ent)) + list(grafo.predecessors(ent))
            nodos_a_explorar = [ent] + vecinos
            
            # 3. Asignar puntuación basada en el número de relaciones relevantes
            for nodo in nodos_a_explorar:
                # Extraer el chunk_id de las aristas que tocan este nodo
                for origen, destino, datos_arista in grafo.edges(nodo, data=True):
                    chunk_id = datos_arista.get("chunk_id")
                    if chunk_id:
                        frecuencia_chunks[chunk_id] = frecuencia_chunks.get(chunk_id, 0) + 1
                        
    # 4. Ordenar chunks por número de relaciones (mayor relevancia primero)
    chunks_ordenados = sorted(frecuencia_chunks.items(), key=lambda x: x[1], reverse=True)
    
    # Devolver solo la lista ordenada de chunk_ids
    return [chunk for chunk, frec in chunks_ordenados]

def procesar_consultas(index, metadata, consultas_texto, model, ner_pipeline, grafo):
    lista_de_resultados = []
    
    # Crear un diccionario inverso rápido: chunk_id -> faiss_id para recuperar textos rápidamente
    chunk_to_faiss = {info["chunk_id"]: idx for idx, info in metadata.items()}
    
    for q_id, texto_consulta in consultas_texto.items():
        # A. Recuperación Vectorial (FAISS)
        vector_consulta = codificar_texto(texto_consulta, model)
        # Ampliamos a k=60 para tener una buena intersección con el grafo antes de cortar a 10
        distancias, indices_faiss = index.search(vector_consulta, k=60) 
        
        # B. Recuperación Grafo de Conocimiento
        candidatos_grafo = obtener_candidatos_grafo(texto_consulta, ner_pipeline, grafo)
        
        # C. Fusión RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        k0 = 60 # Constante de suavizado sugerida
        
        # 1. Puntuar candidatos de FAISS
        for rank, idx in enumerate(indices_faiss[0]):
            if idx == -1: continue # FAISS devuelve -1 si no hay suficientes vectores
            idx_clave = str(idx)
            chunk_id = metadata[idx_clave]["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k0 + rank + 1))
            
        # 2. Puntuar candidatos del Grafo
        for rank, chunk_id in enumerate(candidatos_grafo):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k0 + rank + 1))
            
        # 3. Ordenar todos los chunks por su puntaje combinado RRF
        chunks_finales = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # D. Construir formato final aplicando Max Pooling
        fragmentos_encontrados = []
        scores_documentos = {} 
        
        for chunk_id, score_rrf in chunks_finales:
            faiss_idx = chunk_to_faiss.get(chunk_id)
            if not faiss_idx: continue # Prevención de errores si el grafo tiene un chunk inexistente
            
            fragmento_info = metadata[faiss_idx]
            doc_id = fragmento_info["doc_id"]
            
            # --- Agregación a nivel de documento (Max Pooling) usando el Score RRF ---
            if doc_id not in scores_documentos:
                scores_documentos[doc_id] = score_rrf
            else:
                if score_rrf > scores_documentos[doc_id]:
                    scores_documentos[doc_id] = score_rrf
                    
            # --- Almacenar el TOP 10 de fragmentos ---
            if len(fragmentos_encontrados) < 10:
                texto_fragmento = fragmento_info["text"]
                palabras = texto_fragmento.split()
                
                # Regla estricta: límite 250 palabras
                if len(palabras) > 250:
                    texto_fragmento = " ".join(palabras[:250])
                    
                fragmentos_encontrados.append({
                    "rank": len(fragmentos_encontrados) + 1,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": texto_fragmento
                })
                
        # E. Ordenar Documentos y seleccionar TOP 3
        documentos_ordenados = sorted(scores_documentos.items(), key=lambda item: item[1], reverse=True)
        documentos_encontrados = []
        
        tope_docs = min(3, len(documentos_ordenados)) 
        for i in range(tope_docs):
            documentos_encontrados.append({
                "rank": i + 1,
                "doc_id": documentos_ordenados[i][0]
            })
            
        # F. Empaquetar resultado de la consulta
        resultado = {
            "query_id": q_id,
            "documents": documentos_encontrados,
            "fragments": fragmentos_encontrados
        }
        lista_de_resultados.append(resultado)
        
    return lista_de_resultados

# ---------------------------------------------------------
# 5. EXPORTAR AL FORMATO ESTRICTO JSON LINES (JSONL)
# ---------------------------------------------------------

def guardar_resultados_jsonl(lista_de_resultados, nombre_archivo="resultados.jsonl"):
    print(f"Guardando resultados en {nombre_archivo}...")
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        for resultado in lista_de_resultados:
            # json.dumps asegura que cada objeto sea una línea válida e independiente
            linea_json = json.dumps(resultado, ensure_ascii=False)
            f.write(linea_json + "\n")
    print("¡Archivo generado con éxito!")

# ---------------------------------------------------------
# 6. BLOQUE PRINCIPAL DE EJECUCIÓN
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Configurar modelo y tokenizador
    print("Cargando modelo...")
    # Se carga el modelo una sola vez y se usa tanto para el indexador como para el generador
    nombre_modelo = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
    model = SentenceTransformer(nombre_modelo)

    print("Cargando modelo NER Multilingüe...")
    ner_pipeline = pipeline(
        "ner", 
        model="Babelscape/wikineural-multilingual-ner", 
        aggregation_strategy="simple"
    )

    # 2. Cargar base vectorial y metadatos
    index, metadata = cargar_base_vectorial()
    grafo = cargar_grafo()

    # 3. Cargar las consultas desde el JSON
    consultas_texto = cargar_consultas("consultas.json")
    
    # 4. Procesar las consultas (pasando todos los argumentos necesarios)
    print("Procesando consultas mediante búsqueda híbrida...")
    lista_resultados = procesar_consultas(index, metadata, consultas_texto, model, ner_pipeline, grafo)

# =========================================================
# 7. VISUALIZADOR DE RANKING EN CONSOLA
# =========================================================

print("\n" + "="*50)
print("RESULTADOS DE LA BÚSQUEDA")
print("="*50)

for resultado in lista_resultados:
    print(f"\n Consulta ID: {resultado['query_id']}")
    print(f"Texto: {consultas_texto[resultado['query_id']]}")
    
    print("\n TOP 3 DOCUMENTOS:")
    for doc in resultado['documents']:
        print(f"Rank {doc['rank']}: Documento {doc['doc_id']}")
        
    print("\n TOP FRAGMENTOS (Mostrando los 10 mejores):")
    # Mostramos solo los primeros 3 fragmentos para no saturar la terminal
    for frag in resultado['fragments'][:10]: 
        print(f"Rank {frag['rank']} | {frag['doc_id']} | {frag['chunk_id']}")
        print(f"      Texto: {frag['text'][:90]}...") # Truncamos el texto a 90 caracteres
            
print("\n" + "="*50 + "\n")
# =========================================================

# 5. Guardar los resultados en el formato estricto
guardar_resultados_jsonl(lista_resultados, "resultados.jsonl")

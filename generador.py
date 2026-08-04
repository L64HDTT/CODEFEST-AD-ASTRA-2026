import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch 

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
# 2. CONFIGURACIÓN DEL MODELO 
# --------------------------------------------------------

nombre_modelo = "sentence-transformers/distiluse-base-multilingual-cased-v1" 
model = SentenceTransformer(nombre_modelo)

# ---------------------------------------------------------
# 3. CARGA DE LA BASE VECTORIAL
# ---------------------------------------------------------

def cargar_base_vectorial():
    # Aquí se cargará el índice FAISS que se guardó previamente en la etapa de indexación
    print("Cargando índice FAISS...")
    index = faiss.read_index("base_vectorial/encoder_tu_modelo/index.faiss")
    
    # Aquí cargará la metadata.jsonl en un diccionario para consultarla
    print("Cargando almacén de metadata...")
    metadata = cargar_tu_metadata("base_vectorial/encoder_tu_modelo/metadata.jsonl")
    
    return index, metadata
    pass

# ---------------------------------------------------------
# 4. PROCESAMIENTO DE LAS CONSULTAS
# ---------------------------------------------------------

def codificar_texto(texto, model):
    """
    Convierte el texto de la consulta en un vector numérico normalizado.
    Se espera que 'model' sea una instancia de SentenceTransformer.
    """
    # encode() maneja la tokenización, el pooling, la conversión a NumPy 
    # y la normalización en un solo paso.
    vector_numpy = model.encode([texto], normalize_embeddings=True)
    
    return vector_numpy

def procesar_consultas(index, metadata, consultas_texto, model):
    lista_de_resultados = []
    for q_id, texto_consulta in consultas_texto.items():
        vector_consulta = codificar_texto(texto_consulta, model)
        
        # B. Buscar en FAISS más fragmentos (ej. 30) para asegurar que haya al menos 3 documentos distintos
        distancias, indices_faiss = index.search(vector_consulta, k=30) 
        
        fragmentos_encontrados = []
        scores_documentos = {} 
        
        # C. Construir formato de fragmentos y agrupar para documentos
        for rango, idx in enumerate(indices_faiss[0]):
            idx_clave = str(idx) if isinstance(list(metadata.keys())[0], str) else idx
            fragmento_info = metadata[idx_clave]
            
            doc_id = fragmento_info["doc_id"]
            score_actual = float(distancias[0][rango]) 
            
            # Solo guardamos los primeros 10 para la lista de fragmentos exigida
            if len(fragmentos_encontrados) < 10:
                texto_fragmento = fragmento_info["text"]
                palabras = texto_fragmento.split()
                if len(palabras) > 250:
                    texto_fragmento = " ".join(palabras[:250])
                    print(f"Aviso: Fragmento {fragmento_info['chunk_id']} truncado.")
                
                fragmentos_encontrados.append({
                    "rank": len(fragmentos_encontrados) + 1,
                    "chunk_id": fragmento_info["chunk_id"],
                    "doc_id": doc_id,
                    "text": texto_fragmento
                })
            
            # --- Agregación a nivel de documento (Max Pooling) usando los 30 fragmentos ---
            if doc_id not in scores_documentos:
                scores_documentos[doc_id] = score_actual
            else:
                if score_actual > scores_documentos[doc_id]:
                    scores_documentos[doc_id] = score_actual
                    
        documentos_ordenados = sorted(scores_documentos.items(), key=lambda item: item[1], reverse=True)
        
        # D. Armar la lista de exactamente los 3 documentos más relevantes
        documentos_encontrados = []
        # Si por algún caso extremo hay menos de 3 docs en 30 fragmentos, toma los que haya
        tope_docs = min(3, len(documentos_ordenados)) 
        for i in range(tope_docs):
            documentos_encontrados.append({
                "rank": i + 1,
                "doc_id": documentos_ordenados[i][0]
            })
            
        # E. Armar el resultado final para esta consulta
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
    
    # 2. Cargar base vectorial y metadatos
    index, metadata = cargar_base_vectorial()
    
    # 3. Cargar las consultas desde el JSON
    consultas_texto = cargar_consultas("consultas.json")
    
    # 4. Procesar las consultas (pasando todos los argumentos necesarios)
    print("Procesando consultas y buscando en FAISS...")
    lista_resultados = procesar_consultas(index, metadata, consultas_texto, model)

    # =========================================================
    # VISUALIZADOR DE RANKING EN CONSOLA
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
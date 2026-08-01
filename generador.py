import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch 

# ---------------------------------------------------------
# 0. Carga de las consultas desde un archivo JSON
# ---------------------------------------------------------

def cargar_consultas(ruta_archivo="consultas.json"):
    """Carga las 50 consultas desde un archivo JSON a un diccionario."""
    print(f"Cargando consultas desde {ruta_archivo}...")
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        consultas_texto = json.load(f)
    return consultas_texto

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DEL MODELO (El mismo que usará el arquitecto vectorial)
# ---------------------------------------------------------

nombre_modelo = "distiluse-base-multilingual-cased-v1" # Ejemplo de modelo
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
model = AutoModel.from_pretrained(nombre_modelo)

# ---------------------------------------------------------
# 2. CARGA DE LA BASE VECTORIAL
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
# 3. PROCESAMIENTO DE LAS CONSULTAS
# ---------------------------------------------------------

def codificar_texto(texto):
    """Convierte el texto de la consulta en un vector numérico"""
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extraer el vector usando el estado oculto (mean pooling)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # 1. Convertir el tensor de PyTorch a un arreglo NumPy tipo float32 (FAISS lo requiere)
    vector_numpy = embeddings.detach().cpu().numpy().astype('float32')

    # 2. Normalizar el vector a norma unitaria (L2) para aplicar similitud coseno
    faiss.normalize_L2(vector_numpy)

    return vector_numpy

def procesar_consultas(index, metadata, consultas_texto, tokenizer, model):
    lista_de_resultados = []
    for q_id, texto_consulta in consultas_texto.items():
        # A. Convertir la pregunta en vector
        vector_consulta = codificar_texto(texto_consulta, tokenizer, model)
        
        # B. Buscar en FAISS los 10 fragmentos más similares
        # FAISS devuelve tanto las distancias (scores) como los índices internos
        distancias, indices_faiss = index.search(vector_consulta, k=10)
        
        fragmentos_encontrados = []
        scores_documentos = {} # Diccionario para agrupar puntuaciones por doc_id
        
        # C. Construir formato de fragmentos y agrupar para documentos
        for rango, idx in enumerate(indices_faiss[0]):
            # FAISS devuelve enteros, asegúrate de que tu llave en metadata.jsonl coincida (str o int)
            idx_clave = str(idx) if isinstance(list(metadata.keys())[0], str) else idx
            fragmento_info = metadata[idx_clave]
            
            doc_id = fragmento_info["doc_id"]
            score_actual = float(distancias[0][rango]) # Puntuación de similitud
            
            # --- Regla estricta: Máximo 250 palabras ---
            texto_fragmento = fragmento_info["text"]
            palabras = texto_fragmento.split()
            if len(palabras) > 250:
                texto_fragmento = " ".join(palabras[:250])
                print(f"Aviso: Fragmento {fragmento_info['chunk_id']} excedió las 250 palabras y fue truncado.")
            
            # Agregar a la lista de los 10 fragmentos
            fragmentos_encontrados.append({
                "rank": rango + 1,
                "chunk_id": fragmento_info["chunk_id"],
                "doc_id": doc_id,
                "text": texto_fragmento
            })
            
            # --- Agregación a nivel de documento (Estrategia: Max Pooling) ---
            if doc_id not in scores_documentos:
                scores_documentos[doc_id] = score_actual
            else:
                # Si el documento ya existe, guardamos la puntuación más alta de sus fragmentos
                if score_actual > scores_documentos[doc_id]:
                    scores_documentos[doc_id] = score_actual
                    
        # Ordenar los documentos de mayor a menor según su score agregado
        documentos_ordenados = sorted(scores_documentos.items(), key=lambda item: item[1], reverse=True)
        
        # D. Armar la lista de los 3 documentos más relevantes
        documentos_encontrados = []
        for i, (doc_id, score) in enumerate(documentos_ordenados[:3]):
            documentos_encontrados.append({
                "rank": i + 1,
                "doc_id": doc_id
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
# 4. EXPORTAR AL FORMATO ESTRICTO JSON LINES (JSONL)
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
# 5. BLOQUE PRINCIPAL DE EJECUCIÓN
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Configurar modelo (asegúrate de instanciar tokenizer y model aquí arriba)
    print("Cargando modelo y tokenizador...")
    # tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    # model = AutoModel.from_pretrained(nombre_modelo)
    
    # 2. Cargar base vectorial y metadatos
    # index, metadata = cargar_base_vectorial()
    
    # 3. Cargar las consultas desde el JSON
    consultas_texto = cargar_consultas("consultas.json")
    
    # 4. Procesar las consultas (pasando todos los argumentos necesarios)
    print("Procesando consultas y buscando en FAISS...")
    lista_resultados = procesar_consultas(index, metadata, consultas_texto, tokenizer, model)
    
    # 5. Guardar los resultados en el formato estricto
    guardar_resultados_jsonl(lista_resultados, "resultados.jsonl")
import json
import faiss
import numpy as np

# ---------------------------------------------------------
# 1. CARGA DE LA BASE VECTORIAL
# ---------------------------------------------------------
def cargar_base_vectorial():
    # Aquí cargarás el índice FAISS que guardaste previamente en la etapa de indexación
    print("Cargando índice FAISS...")
    # index = faiss.read_index("base_vectorial/encoder_tu_modelo/index.faiss")
    
    # Aquí cargarás tu metadata.jsonl en un diccionario para consultarla
    print("Cargando almacén de metadata...")
    # metadata = cargar_tu_metadata("base_vectorial/encoder_tu_modelo/metadata.jsonl")
    
    # return index, metadata
    pass

# ---------------------------------------------------------
# 2. PROCESAMIENTO DE LAS CONSULTAS
# ---------------------------------------------------------
def procesar_consultas(index, metadata):
    lista_de_resultados = []
    
    # El reto exige responder a 50 consultas (q001 a q050)
    consultas_id = [f"q{str(i).zfill(3)}" for i in range(1, 51)]
    
    for q_id in consultas_id:
        # Aquí convertirás la consulta de texto a un vector usando tu modelo de Hugging Face
        # vector_consulta = tu_modelo_encoder.encode(texto_de_la_consulta)
        
        # Aquí buscarás en FAISS los fragmentos más cercanos (Similitud Coseno)
        # distancias, identificadores_faiss = index.search(vector_consulta, k=10)
        
        # Aquí construirás las dos listas que exige el formato: 3 documentos y 10 fragmentos
        # (Este es un ejemplo ficticio simulando la estructura exacta del esquema requerido)
        resultado_consulta = {
            "query_id": q_id,
            "documents": [
                {"rank": 1, "doc_id": "DOC-042"},
                {"rank": 2, "doc_id": "DOC-017"},
                {"rank": 3, "doc_id": "DOC-091"}
            ],
            "fragments": [
                {
                    "rank": 1, 
                    "chunk_id": "DOC-042-chunk-007", 
                    "doc_id": "DOC-042", 
                    "text": "Texto del fragmento recuperado (máximo 250 palabras)..."
                }
                # ... Debes agregar hasta completar exactamente 10 fragmentos
            ]
        }
        
        lista_de_resultados.append(resultado_consulta)
        print(f"Consulta {q_id} procesada.")
        
    return lista_de_resultados

# ---------------------------------------------------------
# 3. EXPORTAR AL FORMATO ESTRICTO JSON LINES (JSONL)
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
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ---------------------------------------------------------
if __name__ == "__main__":
    # index, metadata = cargar_base_vectorial()
    # lista_resultados = procesar_consultas(index, metadata)
    
    # Para probar el formato ahora mismo sin el index real, pasamos una lista de prueba vacía:
    lista_de_prueba = [{"query_id": "q001", "documents": [{"rank": 1, "doc_id": "DOC-XYZ"}], "fragments": []}]
    guardar_resultados_jsonl(lista_de_prueba)
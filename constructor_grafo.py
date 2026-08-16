import json
import os
import networkx as nx
from transformers import pipeline
import torch
from tqdm import tqdm  # Importamos la barra de progreso

def procesar_chunk_para_grafo(texto, doc_id, chunk_id, ner_pipeline, grafo):
    """Extrae entidades multilingües y relaciones de un fragmento y las añade al grafo."""
    
    # 1. Ejecutar el modelo NER
    resultados_ner = ner_pipeline(texto)
    
    # 2. Filtrar y limpiar las entidades encontradas (umbral de confianza > 0.7)
    entidades = []
    for ent in resultados_ner:
        if ent['score'] > 0.70:
            entidades.append({
                "text": ent['word'].strip().upper(),
                "label": ent['entity_group'] # Ej: PER, ORG, LOC, MISC
            })
            
    # 3. Eliminar duplicados en el mismo chunk para evitar auto-ciclos
    entidades_unicas = list({v['text']:v for v in entidades}.values())
    
    # 4. Si hay al menos 2 entidades distintas en el chunk, creamos relaciones
    if len(entidades_unicas) >= 2:
        for i in range(len(entidades_unicas)):
            for j in range(i + 1, len(entidades_unicas)):
                ent_origen = entidades_unicas[i]["text"]
                ent_destino = entidades_unicas[j]["text"]
                
                tipo_origen = entidades_unicas[i]["label"]
                tipo_destino = entidades_unicas[j]["label"]
                
                # Añadir nodos con su tipo
                grafo.add_node(ent_origen, tipo=tipo_origen)
                grafo.add_node(ent_destino, tipo=tipo_destino)
                
                # Añadir arista (relación) preservando la metadata obligatoria
                grafo.add_edge(
                    ent_origen, 
                    ent_destino, 
                    relacion="CO_OCURRE_EN_CHUNK", 
                    doc_id=doc_id,
                    chunk_id=chunk_id
                )

def main():
    # --- CONFIGURACIÓN DE PRUEBAS ---
    # Cambia a None cuando quieras procesar TODO el corpus final
    LIMITE_PRUEBAS = None 
    
    # 1. Configurar dispositivo (GPU si está disponible, sino CPU)
    dispositivo = 0 if torch.cuda.is_available() else -1
    motor = "GPU" if dispositivo == 0 else "CPU"
    
    # 2. Cargar el modelo NER multilingüe de Hugging Face
    print(f"Cargando modelo NER multilingüe en {motor} (esto puede tardar la primera vez)...")
    ner_pipeline = pipeline(
        "ner", 
        model="Babelscape/wikineural-multilingual-ner", 
        aggregation_strategy="simple",
        device=dispositivo # Se inyecta el dispositivo aquí
    )
    
    # 3. Inicializar el grafo dirigido
    G = nx.DiGraph()
    
    # 4. Definir la ruta relativa estricta al metadata.jsonl
    ruta_metadata = "base_vectorial/encoder_distiluse-base-multilingual-cased-v1/metadata.jsonl"
    
    if not os.path.exists(ruta_metadata):
        print(f"Error: No se encontró el archivo {ruta_metadata}. Ejecuta el indexador primero.")
        return
        
    # 5. Leer las líneas del archivo
    print(f"Leyendo fragmentos desde {ruta_metadata}...")
    with open(ruta_metadata, 'r', encoding='utf-8') as f:
        lineas = [linea for linea in f if linea.strip()]
        
    # Aplicar límite de pruebas si está configurado
    if LIMITE_PRUEBAS:
        print(f"\n¡MODO DE PRUEBA ACTIVADO! Se procesarán solo {LIMITE_PRUEBAS} fragmentos.")
        lineas = lineas[:LIMITE_PRUEBAS]
    
    # 6. Procesar línea por línea CON BARRA DE PROGRESO (tqdm)
    for linea in tqdm(lineas, desc="Generando Nodos y Aristas NER"):
        chunk_data = json.loads(linea)
        
        # Recuperamos usando .get() para "texto" o "text" por seguridad
        texto = chunk_data.get("texto", chunk_data.get("text", ""))
        doc_id = chunk_data.get("doc_id", "")
        chunk_id = chunk_data.get("chunk_id", "")
        
        # Solo procesar si el chunk tiene los datos obligatorios
        if texto and doc_id and chunk_id:
            procesar_chunk_para_grafo(texto, doc_id, chunk_id, ner_pipeline, G)
                
    # 7. Exportar cumpliendo estrictamente con la ruta y formato del reto
    print(f"\nProcesamiento terminado. Nodos creados: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
    
    ruta_exportacion = "grafo/grafo.graphml"
    os.makedirs(os.path.dirname(ruta_exportacion), exist_ok=True)
    nx.write_graphml(G, ruta_exportacion)
    
    print(f"Grafo exportado exitosamente en: {ruta_exportacion}")

if __name__ == "__main__":
    main()
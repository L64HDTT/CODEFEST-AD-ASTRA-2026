import spacy
import networkx as nx
import os

def procesar_chunk_para_grafo(texto, doc_id, chunk_id, nlp, grafo):
    """Extrae entidades y relaciones de un fragmento y las añade al grafo."""
    doc = nlp(texto)
    
    # Iterar sobre cada oración para buscar entidades que coexisten
    for oracion in doc.sents:
        entidades = oracion.ents
        
        # Si hay al menos 2 entidades en la oración, creamos una relación
        if len(entidades) >= 2:
            for i in range(len(entidades)):
                for j in range(i + 1, len(entidades)):
                    ent_origen = entidades[i].text.strip().upper()
                    ent_destino = entidades[j].text.strip().upper()
                    
                    tipo_origen = entidades[i].label_
                    tipo_destino = entidades[j].label_
                    
                    # Añadir nodos con su tipo (Ej: PER, ORG, LOC)
                    grafo.add_node(ent_origen, tipo=tipo_origen)
                    grafo.add_node(ent_destino, tipo=tipo_destino)
                    
                    # Añadir arista (relación) preservando la metadata obligatoria
                    grafo.add_edge(
                        ent_origen, 
                        ent_destino, 
                        relacion="CO_OCURRE_CON", # Relación base prototipo
                        doc_id=doc_id,
                        chunk_id=chunk_id
                    )

def main():
    # 1. Cargar el modelo NLP de spaCy
    print("Cargando modelo de spaCy...")
    nlp = spacy.load("es_core_news_sm")
    
    # 2. Inicializar el grafo dirigido
    G = nx.DiGraph()
    
    # Simulación de un chunk de tu pipeline (esto vendría de tu metadata.jsonl)
    chunk_simulado = {
        "doc_id": "DOC-001",
        "chunk_id": "DOC-001-chunk-0",
        "text": "La Fuerza Aeroespacial Colombiana implementa Inteligencia Artificial en Bogotá para mejorar la seguridad."
    }
    
    # 3. Procesar el texto
    print("Extrayendo entidades y relaciones...")
    procesar_chunk_para_grafo(
        texto=chunk_simulado["text"],
        doc_id=chunk_simulado["doc_id"],
        chunk_id=chunk_simulado["chunk_id"],
        nlp=nlp,
        grafo=G
    )
    
    # 4. Mostrar información en consola para validación
    print(f"\nGrafo construido con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
    print("Nodos:", G.nodes(data=True))
    print("Aristas:", G.edges(data=True))
    
    # 5. Exportar cumpliendo estrictamente con la ruta y formato del reto
    ruta_exportacion = "grafo/grafo.graphml"
    os.makedirs(os.path.dirname(ruta_exportacion), exist_ok=True)
    nx.write_graphml(G, ruta_exportacion)
    print(f"\nGrafo exportado exitosamente en: {ruta_exportacion}")

if __name__ == "__main__":
    main()  
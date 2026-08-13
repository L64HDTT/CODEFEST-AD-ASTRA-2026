import math
import spacy
from transformers import AutoTokenizer

# 1. Cargar modelos globales
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/distiluse-base-multilingual-cased-v1'
)

nlp_es = spacy.load('es_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')
nlp_de = spacy.load('de_core_news_sm')
nlp_it = spacy.load('it_core_news_sm')
nlp_pt = spacy.load('pt_core_news_sm')

# 2. Filtro de Seguridad Recursivo
def filtrar_fragmentos_seguros(oraciones, tokenizer, max_tokens=250):
    """
    Aísla o elimina oraciones problemáticas antes de hacer el chunking.
    """
    if not oraciones:
        return []

    tokens = [len(tokenizer.encode(o, add_special_tokens=False)) for o in oraciones]

    # FILTRO A: Oración individual > 250 tokens (Se omite)
    for i in range(len(oraciones)):
        if tokens[i] > max_tokens:
            print(f"Advertencia: texto incompatible (oración de {tokens[i]} tokens omitida).")
            fragmento_antes = filtrar_fragmentos_seguros(oraciones[:i], tokenizer, max_tokens)
            fragmento_despues = filtrar_fragmentos_seguros(oraciones[i+1:], tokenizer, max_tokens)
            return fragmento_antes + fragmento_despues

    # FILTRO B: Suma adyacente cruzada > 250 tokens (Se aísla)
    for i in range(1, len(oraciones) - 1):
        suma_anterior = tokens[i-1] + tokens[i]
        suma_siguiente = tokens[i] + tokens[i+1]

        if suma_anterior > max_tokens and suma_siguiente > max_tokens and tokens[i] <= max_tokens:
            chunk_independiente = [oraciones[i]]
            fragmento_antes = filtrar_fragmentos_seguros(oraciones[:i], tokenizer, max_tokens)
            fragmento_despues = filtrar_fragmentos_seguros(oraciones[i+1:], tokenizer, max_tokens)
            return fragmento_antes + [chunk_independiente] + fragmento_despues

    return [oraciones]

# 3. Función Principal de Chunking
def chunkers(data):
    salida = {}
    doc_id = data['doc_id']
    texto_completo = data['texto']

    # Metadatos base
    meta_fuente = data['fuente']
    meta_formato = data['formato']
    meta_fenomeno = data['fenomeno']
    idioma = data['idioma']
    
    parrafos = texto_completo.split('\n')
    q = 0 # Contador global de chunks

    # Selección dinámica del modelo Spacy según el idioma
    if idioma == 'es':
        nlp = nlp_es
    elif idioma == 'en':
        nlp = nlp_en
    elif idioma == 'fr':
        nlp = nlp_fr
    elif idioma == 'de':
        nlp = nlp_de
    elif idioma == 'it':
        nlp = nlp_it
    elif idioma == 'pt':
        nlp = nlp_pt
    else:
        raise ValueError(f'Idioma no soportado por spaCy: {idioma}')

    for parrafo in parrafos:
        if not parrafo.strip():
            continue

        doc = nlp(parrafo)
        oraciones_iniciales = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not oraciones_iniciales:
            continue

        # Pasamos el párrafo por el filtro de seguridad
        bloques_seguros = filtrar_fragmentos_seguros(oraciones_iniciales, tokenizer)

        for bloque in bloques_seguros:
            total_oraciones = len(bloque)
            
            if total_oraciones == 0:
                continue

            usar_ventana_deslizante = False

            # LÓGICA NORMAL (<= 6 oraciones)
            if total_oraciones <= 6:
                texto_chunk = ' '.join(bloque)
                num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False))
                
                # Si a pesar de ser <= 6 oraciones supera los 250 tokens, lo enviamos al backoff
                if num_tokens > 250:
                    usar_ventana_deslizante = True
                else:
                    chunk_id = f'{doc_id}_chunk_{q:03d}'
                    salida[chunk_id] = {
                        'fuente': meta_fuente,
                        'formato': meta_formato,
                        'fenomeno': meta_fenomeno,
                        'text': texto_chunk,
                        'posicion': q,
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'num_tokens': num_tokens,
                        'idioma': idioma # Añadido correctamente
                    }
                    q += 1
            else:
                usar_ventana_deslizante = True

            # LÓGICA AVANZADA (Ventana deslizante con Backoff y Solapamiento)
            if usar_ventana_deslizante:
                OVERLAP = 1
                MAX_TOKENS = 250
                inicio = 0
                
                while inicio < total_oraciones:
                    max_oraciones = 6
                    
                    while max_oraciones > 1:
                        fin = inicio + max_oraciones
                        candidato = bloque[inicio:fin]
                        texto_chunk = ' '.join(candidato)
                        num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False))
                        
                        if num_tokens <= MAX_TOKENS:
                            break
                        else:
                            max_oraciones -= 1 
                            
                    if max_oraciones == 1:
                        fin = inicio + 1
                        candidato = bloque[inicio:fin]
                        texto_chunk = ' '.join(candidato)
                        num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False))
                    
                    chunk_id = f'{doc_id}_chunk_{q:03d}'
                    salida[chunk_id] = {
                        'fuente': meta_fuente,
                        'formato': meta_formato,
                        'fenomeno': meta_fenomeno,
                        'text': texto_chunk,
                        'posicion': q,
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'num_tokens': num_tokens,
                        'idioma': idioma # Añadido correctamente
                    }
                    q += 1
                    
                    oraciones_usadas = len(candidato)
                    avance = oraciones_usadas - OVERLAP
                    
                    if avance < 1:
                        avance = 1
                        
                    inicio += avance

    return salida

# 4. Función Integradora Final
def chunkers_final(data):
    chunks = {}
    for documento in data:
        chunks.update(chunkers(documento))
    return chunks
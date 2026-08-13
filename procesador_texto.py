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
nlp_zh = spacy.load('zh_core_web_sm')


def subdividir_oracion_larga(oracion, tokenizer, max_tokens=200):
    palabras = oracion.split()
    sub_oraciones = []
    bloque_actual = []

    for palabra in palabras:
        bloque_actual.append(palabra)
        candidato = " ".join(bloque_actual) # Usamos truncation=True para evitar warnings durante la medición
        num_tokens = len(tokenizer.encode(candidato, add_special_tokens=False, truncation=True, max_length=512))
        
        if num_tokens >= max_tokens:
            sub_oraciones.append(candidato)
            bloque_actual = []

    if bloque_actual:
        sub_oraciones.append(" ".join(bloque_actual))

    return sub_oraciones


# 2. Filtro de Seguridad Refactorizado
def filtrar_fragmentos_seguros(oraciones, tokenizer, max_tokens=250):
    """
    Procesa oraciones y, si encuentra oraciones gigantes, las sub-divide en 
    lugar de descartarlas.
    """
    if not oraciones:
        return []

    # Se añade truncation=True y max_length=512 para silenciar la advertencia de Transformers
    tokens = [
        len(tokenizer.encode(o, add_special_tokens=False, truncation=True, max_length=512)) 
        for o in oraciones
    ]

    # FILTRO A: Si la oración supera max_tokens, SE SUB-DIVIDE
    for i in range(len(oraciones)):
        if tokens[i] > max_tokens:
            # Sub-dividimos la oración problemática
            sub_oraciones = subdividir_oracion_larga(oraciones[i], tokenizer, max_tokens)
            
            # Reemplazamos la oración larga por sus sub-fragmentos y re-evaluamos recursivamente
            oraciones_nuevas = oraciones[:i] + sub_oraciones + oraciones[i+1:]
            return filtrar_fragmentos_seguros(oraciones_nuevas, tokenizer, max_tokens)

    # FILTRO B: Suma adyacente cruzada > max_tokens
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
    q = 0  # Contador global de chunks

    # Extraer código base
    idioma_base = idioma.split('-')[0].lower() if idioma else 'en'

    # Selección dinámica del modelo Spacy
    if idioma_base == 'es':
        nlp = nlp_es
    elif idioma_base == 'en':
        nlp = nlp_en
    elif idioma_base == 'fr':
        nlp = nlp_fr
    elif idioma_base == 'de':
        nlp = nlp_de
    elif idioma_base == 'it':
        nlp = nlp_it
    elif idioma_base == 'pt':
        nlp = nlp_pt
    elif idioma_base == 'zh':
        nlp = nlp_zh
    else:
        nlp = nlp_en

    for parrafo in parrafos:
        if not parrafo.strip():
            continue

        doc = nlp(parrafo)
        oraciones_iniciales = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not oraciones_iniciales:
            continue

        # Pasamos el párrafo por el filtro seguro
        bloques_seguros = filtrar_fragmentos_seguros(oraciones_iniciales, tokenizer)

        for bloque in bloques_seguros:
            total_oraciones = len(bloque)
            
            if total_oraciones == 0:
                continue

            usar_ventana_deslizante = False

            # LÓGICA NORMAL (<= 6 oraciones)
            if total_oraciones <= 6:
                texto_chunk = ' '.join(bloque)
                num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False, truncation=True, max_length=512))
                
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
                        'idioma': idioma
                    }
                    q += 1
            else:
                usar_ventana_deslizante = True

            # LÓGICA AVANZADA (Ventana deslizante)
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
                        num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False, truncation=True, max_length=512))
                        
                        if num_tokens <= MAX_TOKENS:
                            break
                        else:
                            max_oraciones -= 1 
                            
                    if max_oraciones == 1:
                        fin = inicio + 1
                        candidato = bloque[inicio:fin]
                        texto_chunk = ' '.join(candidato)
                        num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False, truncation=True, max_length=512))
                    
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
                        'idioma': idioma
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
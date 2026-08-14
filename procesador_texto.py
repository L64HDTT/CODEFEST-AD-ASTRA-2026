import math
import spacy
from transformers import AutoTokenizer

# ==========================================================
# 1. CARGA DE MODELOS GLOBALES
# ==========================================================

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


# ==========================================================
# 2. FUNCIONES DE SUBDIVISIÓN Y FILTRADO
# ==========================================================

def subdividir_oracion_larga(oracion, tokenizer, max_tokens=200):
    """
    Sub-divide oraciones gigantes. Maneja palabras normales y también 
    cadenas continuas sin espacios (URLs, tablas pegadas, etc.) para evitar
    recursión infinita.
    """
    palabras = oracion.split()
    
    # CASO BORDE: Si es un solo string gigante sin espacios
    if len(palabras) <= 1:
        paso = max_tokens * 3  # Aprox 3 caracteres por token
        return [oracion[i:i + paso] for i in range(0, len(oracion), paso)]

    sub_oraciones = []
    bloque_actual = []

    for palabra in palabras:
        # Verificar si una sola palabra es gigantesca por sí misma
        num_tokens_palabra = len(tokenizer.encode(palabra, add_special_tokens=False, truncation=True, max_length=512))
        if num_tokens_palabra >= max_tokens:
            if bloque_actual:
                sub_oraciones.append(" ".join(bloque_actual))
                bloque_actual = []
            paso = max_tokens * 3
            sub_oraciones.extend([palabra[k:k + paso] for k in range(0, len(palabra), paso)])
            continue

        candidato = " ".join(bloque_actual + [palabra])
        num_tokens = len(tokenizer.encode(candidato, add_special_tokens=False, truncation=True, max_length=512))

        if num_tokens >= max_tokens:
            if bloque_actual:
                sub_oraciones.append(" ".join(bloque_actual))
            bloque_actual = [palabra]
        else:
            bloque_actual.append(palabra)

    if bloque_actual:
        sub_oraciones.append(" ".join(bloque_actual))

    return sub_oraciones if sub_oraciones else [oracion[:max_tokens * 3]]


def filtrar_fragmentos_seguros(oraciones, tokenizer, max_tokens=250):
    """
    Garantiza que ningún fragmento supere max_tokens sin caer en recursión infinita.
    """
    if not oraciones:
        return []

    oraciones_procesadas = []

    # Iteración iterativa segura sobre oraciones largas
    for o in oraciones:
        num_tokens = len(tokenizer.encode(o, add_special_tokens=False, truncation=True, max_length=512))
        
        if num_tokens > max_tokens:
            sub_frags = subdividir_oracion_larga(o, tokenizer, max_tokens)
            for sf in sub_frags:
                n_tok = len(tokenizer.encode(sf, add_special_tokens=False, truncation=True, max_length=512))
                if n_tok > max_tokens:
                    limite_chars = max_tokens * 3
                    oraciones_procesadas.append(sf[:limite_chars])
                else:
                    oraciones_procesadas.append(sf)
        else:
            oraciones_procesadas.append(o)

    # Evaluación de la regla B (agrupación / aislamiento)
    tokens = [
        len(tokenizer.encode(o, add_special_tokens=False, truncation=True, max_length=512))
        for o in oraciones_procesadas
    ]

    for i in range(1, len(oraciones_procesadas) - 1):
        suma_anterior = tokens[i-1] + tokens[i]
        suma_siguiente = tokens[i] + tokens[i+1]

        if suma_anterior > max_tokens and suma_siguiente > max_tokens and tokens[i] <= max_tokens:
            chunk_independiente = [oraciones_procesadas[i]]
            fragmento_antes = oraciones_procesadas[:i]
            fragmento_despues = oraciones_procesadas[i+1:]
            return [fragmento_antes] + [chunk_independiente] + [fragmento_despues]

    return [oraciones_procesadas]


# ==========================================================
# 3. LÓGICA DE CHUNKING
# ==========================================================

def chunkers(data):
    salida = {}
    doc_id = data['doc_id']
    texto_completo = data['texto']

    meta_fuente = data['fuente']
    meta_formato = data['formato']
    meta_fenomeno = data['fenomeno']
    idioma = data['idioma']
    
    parrafos = texto_completo.split('\n')
    total_parrafos = len(parrafos)
    q = 0  # Contador global de chunks

    idioma_base = idioma.split('-')[0].lower() if idioma else 'en'

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

    print(f"\n[INICIO DOC] {doc_id} | Total párrafos a procesar: {total_parrafos}", flush=True)

    for i, parrafo in enumerate(parrafos):
        parrafo_clean = parrafo.strip()
        if not parrafo_clean:
            continue

        len_chars = len(parrafo_clean)
        
        # --- PUNTO DE CONTROL 1: Entrada a SpaCy ---
        print(f"  --> [Párrafo {i+1}/{total_parrafos}] Longitud: {len_chars} caracteres. Evaluando con SpaCy...", end="", flush=True)

        # SI SE CONGELA AQUÍ: El problema es el parser sintáctico de SpaCy con un párrafo gigante o mal formateado
        doc = nlp(parrafo_clean[:15000])  # Truncado de seguridad para evitar cuelgues sintácticos
        oraciones_iniciales = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        print(f" Completado ({len(oraciones_iniciales)} oraciones).", flush=True)

        if not oraciones_iniciales:
            continue

        # --- PUNTO DE CONTROL 2: Filtro de seguridad ---
        print(f"      [Filtro] Ejecutando 'filtrar_fragmentos_seguros'...", end="", flush=True)
        bloques_seguros = filtrar_fragmentos_seguros(oraciones_iniciales, tokenizer)
        print(f" OK (Bloques generados: {len(bloques_seguros)}).", flush=True)

        for idx_b, bloque in enumerate(bloques_seguros):
            total_oraciones = len(bloque)
            if total_oraciones == 0:
                continue

            usar_ventana_deslizante = False

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

            # --- PUNTO DE CONTROL 3: Ventana Deslizante ---
            if usar_ventana_deslizante:
                print(f"      [Ventana Deslizante] Procesando bloque {idx_b+1}/{len(bloques_seguros)} ({total_oraciones} oraciones)...", end="", flush=True)
                OVERLAP = 1
                MAX_TOKENS = 250
                inicio = 0
                
                iteraciones_seguridad = 0
                max_iteraciones_permitidas = total_oraciones * 10

                while inicio < total_oraciones:
                    iteraciones_seguridad += 1
                    if iteraciones_seguridad > max_iteraciones_permitidas:
                        print(f"\n[ERROR CRÍTICO] Bucle infinito detectado en Ventana Deslizante para doc {doc_id}, párrafo {i+1}!", flush=True)
                        break

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

                print(" OK.", flush=True)

    print(f"[FIN DOC] {doc_id} | Chunks creados: {q}\n", flush=True)
    return salida

# ==========================================================
# 4. FUNCIÓN INTEGRADORA FINAL
# ==========================================================

def chunkers_final(data):
    chunks = {}
    for documento in data:
        chunks.update(chunkers(documento))
    return chunks
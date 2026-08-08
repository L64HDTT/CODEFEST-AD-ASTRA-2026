import math
import spacy
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/distiluse-base-multilingual-cased-v1'
)

nlp_es = spacy.load('es_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')
nlp_de = spacy.load('de_core_news_sm')
nlp_it = spacy.load('it_core_news_sm')
nlp_pt = spacy.load('pt_core_news_sm')

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
        oraciones = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        total_oraciones = len(oraciones)

        if total_oraciones == 0:
            continue
        if total_oraciones <= 6 and num_tokens < 250:
            texto_chunk = ' '.join(oraciones)
            num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False))
            chunk_id = f'{doc_id}_chunk_{q:03d}'
            salida[chunk_id] = {'fuente': meta_fuente,'formato': meta_formato,'fenomeno': meta_fenomeno,'texto': texto_chunk,'posicion': q,'doc_id': doc_id,'chunk_id': chunk_id,'num_tokens': num_tokens,'idioma': idioma}
            q += 1

        else:
            l = math.ceil(total_oraciones / 6)
            b = total_oraciones // l
            r = total_oraciones % l

            for j in range(l):
                if j < r:
                    inicio = j * (b + 1)
                    fin = inicio + (b + 1)
                else:
                    inicio = r * (b + 1) + (j - r) * b
                    fin = inicio + b

                texto_chunk = ' '.join(oraciones[inicio:fin])
                num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=False))

                # ¡Corregido! Se usa doc_id en lugar de doc_id_num
                chunk_id = f'{doc_id}_chunk_{q:03d}'
                salida[chunk_id] = {'fuente': meta_fuente,'formato': meta_formato,'fenomeno': meta_fenomeno,'texto': texto_chunk,'posicion': q,'doc_id': doc_id,'chunk_id': chunk_id,'num_tokens': num_tokens}
                if num_tokens > 250:
                    print(f'Advertencia: {chunk_id} tiene {num_tokens} tokens.')
                q += 1

    return salida
def chunkers_final(data):
    '''
    Recibe la lista de documentos y retorna un diccionario único
    con todos los chunks.
    '''
    chunks = {}
    for documento in data:
        chunks.update(chunkers(documento))
    return chunks
dic_chunks=chunkers_final(documentos)
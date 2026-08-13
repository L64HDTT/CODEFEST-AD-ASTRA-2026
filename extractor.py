import os
import re
import json

import pymupdf
import pandas as pd
import pytesseract

from bs4 import BeautifulSoup
from PIL import Image
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


# ==========================================================
# EXTRACCIÓN PDF
# ==========================================================

def extraer_pdf(ruta, limite_caracteres_ocr=50):
    """
    Extrae el texto de un PDF manteniendo los indicadores de página.
    Si una página contiene menos caracteres de los especificados, aplica OCR.
    """
    texto_paginas = []
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta

    with pymupdf.open(ruta_limpia) as pdf:
        for num_pagina, pagina in enumerate(pdf):
            texto_nativo = pagina.get_text("text")
            
            # Verificar si la página necesita OCR (escaneada o imagen)
            if len(texto_nativo.strip()) < limite_caracteres_ocr: # Renderizar página a imagen de alta resolución
                pix = pagina.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Ejecutar OCR focalizado
                texto_ocr = pytesseract.image_to_string(img, lang="spa+eng")
                texto_final = texto_ocr if texto_ocr.strip() else texto_nativo
            else:
                texto_final = texto_nativo
            
            # Agregar encabezado de página para mantener trazabilidad
            encabezado = f"--- Página {num_pagina + 1} ---\n"
            texto_paginas.append(encabezado + texto_final)

    return "\n\n".join(texto_paginas)


# ==========================================================
# EXTRACCIÓN HTML
# ==========================================================

def extraer_html(ruta):
    """
    Extrae únicamente el texto visible de un HTML.
    """
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta

    with open(ruta_limpia, "r", encoding="utf-8", errors="ignore") as archivo:
        html = archivo.read()

    soup = BeautifulSoup(html, "html.parser")

    for elemento in soup(["script", "style"]):
        elemento.extract()

    return soup.get_text(separator="\n")


# ==========================================================
# EXTRACCIÓN JSON
# ==========================================================

def recorrer_json(obj):
    """
    Recorre cualquier estructura JSON extrayendo texto y conservando URLs 
    de fuentes/metadatos sin alterar el flujo.
    """
    texto = ""

    if isinstance(obj, dict):
        for clave, valor in obj.items():
            texto += recorrer_json(valor)

    elif isinstance(obj, list):
        for elemento in obj:
            texto += recorrer_json(elemento)

    elif isinstance(obj, str):
        texto += obj + "\n"

    elif isinstance(obj, (int, float, bool)):
        texto += str(obj) + "\n"

    return texto


def extraer_json(ruta):
    """
    Extrae el contenido textual de un JSON.
    """
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta

    with open(ruta_limpia, "r", encoding="utf-8", errors="ignore") as archivo:
        datos = json.load(archivo)

    return recorrer_json(datos)


# ==========================================================
# EXTRACCIÓN CSV (OPTIMIZADO PARA EVITAR CONGELAMIENTOS)
# ==========================================================

def extraer_csv(ruta):
    """
    Convierte un CSV a texto en formato registro por registro, evitando 
    la generación de volúmenes masivos de texto plano inútil.
    """
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta
    
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(
                ruta_limpia, 
                sep=sep, 
                engine='c', 
                on_bad_lines='skip', 
                encoding_errors='ignore',
                low_memory=False
            )
            
            lineas = []
            cols = df.columns.tolist()
            # Se procesan las filas en formato 'Columna: Valor' delimitado
            for row in df.head(1000).itertuples(index=False):
                registro = [f"{col}: {val}" for col, val in zip(cols, row) if pd.notna(val)]
                if registro:
                    lineas.append(" | ".join(registro))
                
            return "\n".join(lineas)
        except Exception:
            continue

    # Fallback de lectura simple
    try:
        with open(ruta_limpia, "r", encoding="utf-8", errors="ignore") as f:
            lineas = [f.readline() for _ in range(1000)]
            return "".join(lineas)
    except Exception:
        return ""


# ==========================================================
# EXTRACCIÓN XLSX
# ==========================================================

def extraer_xlsx(ruta):
    """
    Convierte un archivo Excel a texto.
    """
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta
    try:
        df = pd.read_excel(ruta_limpia)
        lineas = []
        cols = df.columns.tolist()
        for row in df.head(1000).itertuples(index=False):
            registro = [f"{col}: {val}" for col, val in zip(cols, row) if pd.notna(val)]
            if registro:
                lineas.append(" | ".join(registro))
        return "\n".join(lineas)
    except Exception:
        return ""


# ==========================================================
# EXTRACCIÓN IMÁGENES (OCR / LINEAMIENTO SIN INVENTAR TEXTO)
# ==========================================================

def extraer_imagen(ruta):
    """
    Aplica OCR a las imágenes. Si la imagen no posee texto útil,
    devuelve una cadena vacía conservando únicamente sus metadatos.
    """
    try:
        ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta
        imagen = Image.open(ruta_limpia)
        texto = pytesseract.image_to_string(imagen, lang="spa+eng")
        return texto.strip()
    except Exception:
        # En caso de error o ausencia de motor OCR, retorna texto vacío
        return ""


# ==========================================================
# EXTRACCIÓN PBF
# ==========================================================

def extraer_pbf(ruta):
    return "Extracción PBF no implementada aún. El archivo fue detectado correctamente."


# ==========================================================
# LIMPIEZA DEL TEXTO
# ==========================================================

def limpiar_texto(texto):
    """
    Limpia el texto extraído conservando la estructura de párrafos.
    """
    if not texto:
        return ""

    texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', texto)
    texto = re.sub(r'[\uFFF0-\uFFFF]', '', texto)
    texto = re.sub(r'(?m)^\s*\d+\s*$', '', texto)
    texto = re.sub(r'[ \t]+$', '', texto, flags=re.MULTILINE)
    texto = re.sub(r' +', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    return texto.strip()


# ==========================================================
# DETECCIÓN DE IDIOMA
# ==========================================================

def identificar_idioma(texto):
    """
    Detecta el idioma del texto. Devuelve 'es' si está vacío o no se identifica.
    """
    if not texto or len(texto.strip()) == 0:
        return "es"

    try:
        return detect(texto[:5000])
    except Exception:
        return "es"


# ==========================================================
# SELECCIÓN AUTOMÁTICA DEL EXTRACTOR
# ==========================================================

def extraer_texto(ruta):
    extension = os.path.splitext(ruta)[1].lower()

    if extension == ".pdf":
        return extraer_pdf(ruta)
    elif extension in [".html", ".htm"]:
        return extraer_html(ruta)
    elif extension == ".json":
        return extraer_json(ruta)
    elif extension == ".csv":
        return extraer_csv(ruta)
    elif extension == ".xlsx":
        return extraer_xlsx(ruta)
    elif extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        return extraer_imagen(ruta)
    elif extension == ".pbf":
        return extraer_pbf(ruta)
    else:
        raise ValueError(f"Formato no soportado: {extension}")


# ==========================================================
# PROCESAR UN DOCUMENTO
# ==========================================================

def procesar_documento(ruta_archivo, doc_id, fenomeno):
    """
    Procesa un documento de cualquier formato soportado.
    Si una imagen no contiene texto, se conserva con texto vació manteniendo su metadata.
    """
    try:
        nombre_base = os.path.basename(ruta_archivo)
        print(f"Procesando: {nombre_base}")

        texto = extraer_texto(ruta_archivo)
        texto = limpiar_texto(texto)
        idioma = identificar_idioma(texto)
        formato = os.path.splitext(ruta_archivo)[1][1:].lower()

        documento = {
            "doc_id": doc_id,
            "fuente": nombre_base,
            "formato": formato,
            "fenomeno": fenomeno,
            "idioma": idioma,
            "texto": texto
        }

        return documento

    except Exception as e:
        print(f"\nError procesando {ruta_archivo}: {e}")
        return None


# ==========================================================
# PROCESAR TODOS LOS DOCUMENTOS DE UNA CARPETA O ARCHIVO
# ==========================================================

def procesar_carpeta(carpeta, fenomeno=1):
    """
    Procesa todos los archivos soportados ignorando entornos virtuales o temporales.
    """
    documentos = []
    formatos = (
        ".pdf", ".html", ".htm", ".json", ".csv", ".xlsx", 
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pbf"
    )

    CARPETAS_IGNORADAS = {
        '.venv', 'venv', 'env', '.git', '__pycache__', 
        'node_modules', '.idea', '.vscode', 'entrega', 'build'
    }

    ARCHIVOS_IGNORADOS = {
        'consultas.json', 'texto_extraido.json', 'metadata.jsonl', 'resultados.jsonl'
    }

    contador = 1

    if os.path.isfile(carpeta):
        if carpeta.lower().endswith(formatos):
            ruta_abs = os.path.abspath(carpeta)
            doc_id = f"DOC-{contador:03d}"
            doc = procesar_documento(ruta_abs, doc_id, fenomeno)
            if doc is not None:
                documentos.append(doc)
        return documentos

    for raiz, directorios, archivos in os.walk(carpeta):
        directorios[:] = [d for d in directorios if d.lower() not in CARPETAS_IGNORADAS and not d.startswith('.')]

        for archivo in sorted(archivos):
            nombre_lower = archivo.lower()

            if nombre_lower in ARCHIVOS_IGNORADOS:
                continue

            if nombre_lower.endswith(formatos):
                ruta = os.path.join(raiz, archivo)
                ruta = os.path.abspath(ruta)

                if os.name == 'nt':
                    ruta = f"\\\\?\\{ruta}"

                doc_id = f"DOC-{contador:03d}"
                
                documento = procesar_documento(
                    ruta_archivo=ruta,
                    doc_id=doc_id,
                    fenomeno=fenomeno
                )

                if documento is not None:
                    documentos.append(documento)
                    contador += 1

    return documentos


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

if __name__ == "__main__":

    carpeta = "datos"
    archivo_salida = "datos/texto_extraido.json"
    fenomeno = 1

    if not os.path.exists(carpeta):
        print(f"La ruta '{carpeta}' no existe.")
    else:
        print("Iniciando extracción...")
        documentos = procesar_carpeta(carpeta, fenomeno)

        print("\n" + "=" * 70)
        print(f"Se procesaron {len(documentos)} documento(s).")
        print("=" * 70)

        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        with open(archivo_salida, "w", encoding="utf-8") as archivo_json:
            json.dump(documentos, archivo_json, ensure_ascii=False, indent=4)
            
        print("¡Extracción completada exitosamente!")
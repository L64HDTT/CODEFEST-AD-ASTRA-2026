import os
import re
import json

import fitz  # PyMuPDF
import pandas as pd
import pytesseract

from bs4 import BeautifulSoup
from PIL import Image
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


# ==========================================================
# EXTRACCIÓN PDF
# ==========================================================

def extraer_pdf(ruta):
    """
    Extrae el texto de un archivo PDF utilizando PyMuPDF.
    Garantiza una lectura rápida extrayendo únicamente el texto vectorizado.
    """
    texto = ""
    # Remover prefijo de ruta larga para compatibilidad
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta

    with fitz.open(ruta_limpia) as pdf:
        for pagina in pdf:
            texto += pagina.get_text("text")
            texto += "\n\n"

    return texto


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

    # Eliminar scripts y estilos
    for elemento in soup(["script", "style"]):
        elemento.extract()

    texto = soup.get_text(separator="\n")

    return texto


# ==========================================================
# EXTRACCIÓN JSON
# ==========================================================

def recorrer_json(obj):
    """
    Recorre cualquier estructura JSON y obtiene únicamente
    el contenido textual.
    """
    texto = ""

    if isinstance(obj, dict):
        for valor in obj.values():
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
# EXTRACCIÓN CSV (OPTIMIZADO)
# ==========================================================

def extraer_csv(ruta):
    """
    Convierte un CSV a texto de forma ultra-rápida y resistente.
    """
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta
    try:
        # Uso vectorizado con pandas para evitar iteraciones lentas por fila/columna
        df = pd.read_csv(
            ruta_limpia, 
            sep=None, 
            engine='python', 
            on_bad_lines='skip', 
            encoding_errors='ignore'
        )
        return df.to_string(index=False)

    except Exception:
        import csv
        texto = ""
        with open(ruta_limpia, "r", encoding="utf-8", errors="ignore") as f:
            lector = csv.reader(f)
            try:
                encabezados = next(lector)
            except StopIteration:
                return ""
            
            for fila in lector:
                linea = [f"{encabezados[i]}: {fila[i]}" for i in range(min(len(fila), len(encabezados)))]
                texto += " | ".join(linea) + "\n"

        return texto


# ==========================================================
# EXTRACCIÓN XLSX
# ==========================================================

def extraer_xlsx(ruta):
    """
    Convierte un archivo Excel a texto de forma optimizada.
    """
    ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta
    try:
        df = pd.read_excel(ruta_limpia)
        return df.to_string(index=False)
    except Exception:
        return ""


# ==========================================================
# EXTRACCIÓN IMÁGENES (OCR / CONTROL DE ERRORES)
# ==========================================================

def extraer_imagen(ruta):
    """
    Extrae texto mediante OCR protegiendo la ejecución ante errores
    o falta de Tesseract en el sistema.
    """
    try:
        ruta_limpia = ruta.replace("\\\\?\\", "") if ruta.startswith("\\\\?\\") else ruta
        imagen = Image.open(ruta_limpia)
        texto = pytesseract.image_to_string(imagen, lang="spa+eng")
        return texto
    except Exception as e:
        # Evita interrumpir la ejecución completa del corpus
        return ""


# ==========================================================
# EXTRACCIÓN PBF
# ==========================================================

def extraer_pbf(ruta):
    """
    Función preparada para archivos PBF.
    """
    return (
        "Extracción PBF no implementada aún. "
        "El archivo fue detectado correctamente."
    )


# ==========================================================
# LIMPIEZA DEL TEXTO
# ==========================================================

def limpiar_texto(texto):
    """
    Limpia el texto extraído de cualquier formato.
    """
    if not texto:
        return ""

    # Eliminar caracteres de control
    texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', texto)

    # Eliminar caracteres Unicode invisibles
    texto = re.sub(r'[\uFFF0-\uFFFF]', '', texto)

    # Eliminar líneas que solo contienen números
    texto = re.sub(r'(?m)^\s*\d+\s*$', '', texto)

    # Eliminar espacios al final de línea
    texto = re.sub(r'[ \t]+$', '', texto, flags=re.MULTILINE)

    # Reemplazar múltiples espacios por uno
    texto = re.sub(r' +', ' ', texto)

    # Reducir muchos saltos de línea
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    # Normalizar UTF-8
    texto = texto.encode("utf-8", "ignore").decode("utf-8")

    return texto.strip()


# ==========================================================
# DETECCIÓN DE IDIOMA
# ==========================================================

def identificar_idioma(texto):
    """
    Detecta el idioma predominante del documento.
    """
    if len(texto.strip()) == 0:
        return "es"

    try:
        idioma = detect(texto[:5000])
        return idioma
    except Exception:
        return "es"


# ==========================================================
# SELECCIÓN AUTOMÁTICA DEL EXTRACTOR
# ==========================================================

def extraer_texto(ruta):
    """
    Detecta la extensión del archivo y utiliza el extractor correspondiente.
    """
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
    Retorna un diccionario listo para el módulo de chunking.
    """
    try:
        nombre_base = os.path.basename(ruta_archivo)
        print(f"Procesando: {nombre_base}")

        # Extraer texto según el tipo de archivo
        texto = extraer_texto(ruta_archivo)

        # Limpiar texto
        texto = limpiar_texto(texto)

        # Detectar idioma
        idioma = identificar_idioma(texto)

        # Obtener formato
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
    Procesa todos los archivos soportados de una carpeta/subcarpetas o un archivo individual.
    """
    documentos = []
    formatos = (
        ".pdf", ".html", ".htm", ".json", ".csv", ".xlsx", 
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pbf"
    )

    contador = 1

    # Caso 1: Si la ruta dada es un archivo directo
    if os.path.isfile(carpeta):
        if carpeta.lower().endswith(formatos):
            ruta_abs = os.path.abspath(carpeta)
            doc_id = f"DOC-{contador:03d}"
            doc = procesar_documento(ruta_abs, doc_id, fenomeno)
            if doc is not None:
                documentos.append(doc)
        return documentos

    # Caso 2: Recorrido de carpeta y subcarpetas
    for raiz, directorios, archivos in os.walk(carpeta):
        for archivo in sorted(archivos):
            if archivo.lower().endswith(formatos):
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
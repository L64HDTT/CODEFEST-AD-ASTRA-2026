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
    """

    texto = ""

    with fitz.open(ruta) as pdf:

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

    with open(ruta, "r", encoding="utf-8") as archivo:

        html = archivo.read()

    soup = BeautifulSoup(html, "html.parser")

    # eliminar scripts y estilos

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

    with open(ruta, "r", encoding="utf-8") as archivo:

        datos = json.load(archivo)

    return recorrer_json(datos)


# ==========================================================
# EXTRACCIÓN CSV
# ==========================================================

def extraer_csv(ruta):
    """
    Convierte un CSV a texto preservando el nombre
    de las columnas.
    """

    df = pd.read_csv(ruta)

    texto = ""

    for _, fila in df.iterrows():

        for columna in df.columns:

            texto += f"{columna}: {fila[columna]}\n"

        texto += "\n"

    return texto


# ==========================================================
# EXTRACCIÓN XLSX
# ==========================================================

def extraer_xlsx(ruta):
    """
    Convierte un archivo Excel a texto.
    """

    df = pd.read_excel(ruta)

    texto = ""

    for _, fila in df.iterrows():

        for columna in df.columns:

            texto += f"{columna}: {fila[columna]}\n"

        texto += "\n"

    return texto


# ==========================================================
# EXTRACCIÓN IMÁGENES (OCR)
# ==========================================================

def extraer_imagen(ruta):
    """
    Extrae texto mediante OCR.
    Requiere tener instalado Tesseract OCR.
    """

    imagen = Image.open(ruta)

    texto = pytesseract.image_to_string(
        imagen,
        lang="spa+eng"
    )

    return texto


# ==========================================================
# EXTRACCIÓN PBF
# ==========================================================

def extraer_pbf(ruta):
    """
    Función preparada para archivos PBF.

    Puede ampliarse posteriormente utilizando pyosmium.
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

    - Elimina caracteres de control.
    - Elimina caracteres Unicode invisibles.
    - Elimina números de página aislados.
    - Conserva los párrafos.
    - Reduce espacios redundantes.
    """

    if not texto:
        return ""

    # Eliminar caracteres de control
    texto = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',
        '',
        texto
    )

    # Eliminar caracteres Unicode invisibles
    texto = re.sub(
        r'[\uFFF0-\uFFFF]',
        '',
        texto
    )

    # Eliminar líneas que solo contienen números
    texto = re.sub(
        r'(?m)^\s*\d+\s*$',
        '',
        texto
    )

    # Eliminar espacios al final de línea
    texto = re.sub(
        r'[ \t]+$',
        '',
        texto,
        flags=re.MULTILINE
    )

    # Reemplazar múltiples espacios por uno
    texto = re.sub(
        r' +',
        ' ',
        texto
    )

    # Reducir muchos saltos de línea
    texto = re.sub(
        r'\n{3,}',
        '\n\n',
        texto
    )

    # Normalizar UTF-8
    texto = texto.encode(
        "utf-8",
        "ignore"
    ).decode(
        "utf-8"
    )

    return texto.strip()


# ==========================================================
# DETECCIÓN DE IDIOMA
# ==========================================================

def identificar_idioma(texto):
    """
    Detecta el idioma predominante del documento.
    """

    if len(texto.strip()) == 0:
        return "desconocido"

    try:

        idioma = detect(texto[:5000])

        return idioma

    except Exception:

        return "desconocido"


# ==========================================================
# SELECCIÓN AUTOMÁTICA DEL EXTRACTOR
# ==========================================================

def extraer_texto(ruta):
    """
    Detecta la extensión del archivo y utiliza
    el extractor correspondiente.
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

    elif extension in [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    ]:
        return extraer_imagen(ruta)

    elif extension == ".pbf":
        return extraer_pbf(ruta)

    else:

        raise ValueError(
            f"Formato no soportado: {extension}"
        )
# ==========================================================
# LIMPIEZA DEL TEXTO
# ==========================================================

def limpiar_texto(texto):
    """
    Limpia el texto extraído de cualquier formato.

    - Elimina caracteres de control.
    - Elimina caracteres Unicode invisibles.
    - Elimina números de página aislados.
    - Conserva los párrafos.
    - Reduce espacios redundantes.
    """

    if not texto:
        return ""

    # Eliminar caracteres de control
    texto = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',
        '',
        texto
    )

    # Eliminar caracteres Unicode invisibles
    texto = re.sub(
        r'[\uFFF0-\uFFFF]',
        '',
        texto
    )

    # Eliminar líneas que solo contienen números
    texto = re.sub(
        r'(?m)^\s*\d+\s*$',
        '',
        texto
    )

    # Eliminar espacios al final de línea
    texto = re.sub(
        r'[ \t]+$',
        '',
        texto,
        flags=re.MULTILINE
    )

    # Reemplazar múltiples espacios por uno
    texto = re.sub(
        r' +',
        ' ',
        texto
    )

    # Reducir muchos saltos de línea
    texto = re.sub(
        r'\n{3,}',
        '\n\n',
        texto
    )

    # Normalizar UTF-8
    texto = texto.encode(
        "utf-8",
        "ignore"
    ).decode(
        "utf-8"
    )

    return texto.strip()


# ==========================================================
# DETECCIÓN DE IDIOMA
# ==========================================================

def identificar_idioma(texto):
    """
    Detecta el idioma predominante del documento.
    """

    if len(texto.strip()) == 0:
        return "desconocido"

    try:

        idioma = detect(texto[:5000])

        return idioma

    except Exception:

        return "desconocido"


# ==========================================================
# SELECCIÓN AUTOMÁTICA DEL EXTRACTOR
# ==========================================================

def extraer_texto(ruta):
    """
    Detecta la extensión del archivo y utiliza
    el extractor correspondiente.
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

    elif extension in [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    ]:
        return extraer_imagen(ruta)

    elif extension == ".pbf":
        return extraer_pbf(ruta)

    else:

        raise ValueError(
            f"Formato no soportado: {extension}"
        )
# ==========================================================
# PROCESAR UN DOCUMENTO
# ==========================================================

def procesar_documento(ruta_archivo, doc_id, fenomeno):
    """
    Procesa un documento de cualquier formato soportado.

    Retorna un diccionario listo para el módulo de chunking.
    """

    try:

        print(f"Procesando: {os.path.basename(ruta_archivo)}")

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
            "fuente": os.path.basename(ruta_archivo),
            "formato": formato,
            "fenomeno": fenomeno,
            "idioma": idioma,
            "texto": texto

        }

        return documento

    except Exception as e:

        print(f"\nError procesando {ruta_archivo}")
        print(e)

        return None


# ==========================================================
# PROCESAR TODOS LOS DOCUMENTOS DE UNA CARPETA
# ==========================================================

def procesar_carpeta(carpeta, fenomeno):
    """
    Procesa todos los archivos soportados de una carpeta.

    Genera automáticamente:

        DOC-001
        DOC-002
        DOC-003
        ...

    Devuelve una lista de documentos.
    """

    documentos = []

    formatos = (

        ".pdf",
        ".html",
        ".htm",
        ".json",
        ".csv",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".pbf"

    )

    archivos = sorted(os.listdir(carpeta))

    contador = 1

    for archivo in archivos:

        if archivo.lower().endswith(formatos):

            ruta = os.path.join(carpeta, archivo)

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

    fenomeno = 1

    if not os.path.exists(carpeta):

        print(f"La carpeta '{carpeta}' no existe.")
        print("Créala dentro del proyecto y coloca allí los archivos.")

    else:

        documentos = procesar_carpeta(carpeta, fenomeno)

        print("\n")
        print("=" * 70)
        print(f"Se procesaron {len(documentos)} documento(s).")
        print("=" * 70)

        for doc in documentos:

            print("\n")

            print("=" * 70)

            print(f"Documento : {doc['doc_id']}")
            print(f"Fuente    : {doc['fuente']}")
            print(f"Formato   : {doc['formato']}")
            print(f"Fenómeno  : {doc['fenomeno']}")
            print(f"Idioma    : {doc['idioma']}")

            print("\nPrimeros 500 caracteres:\n")

            print(doc["texto"][:500])

            print("\nLongitud del texto:")

            print(len(doc["texto"]))

            print("=" * 70)

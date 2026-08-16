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
    Aplica un recorte (clip) para ignorar el 8% superior e inferior,
    evitando extraer encabezados y pies de página repetitivos.
    """
    texto = ""
    with fitz.open(ruta) as pdf:
        for pagina in pdf:
            # Obtener dimensiones totales de la página
            rect = pagina.rect
            margen_y = rect.height * 0.08
            
            # Crear un rectángulo de lectura que omite top/bottom
            clip_rect = fitz.Rect(rect.x0, rect.y0 + margen_y, rect.x1, rect.y1 - margen_y)
            
            # Extraer solo lo que esté dentro de ese rectángulo
            texto += pagina.get_text("text", clip=clip_rect)
            texto += "\n\n"
            
    return texto

# ==========================================================
# EXTRACCIÓN HTML
# ==========================================================

def extraer_html(ruta):
    """
    Extrae únicamente el texto visible y útil de un HTML.
    Elimina scripts, estilos, menús de navegación, cabeceras y pies de página.
    """
    with open(ruta, "r", encoding="utf-8") as archivo:
        html = archivo.read()

    soup = BeautifulSoup(html, "html.parser")

    # Eliminar elementos que generan "ruido semántico"
    etiquetas_basura = ["script", "style", "nav", "footer", "header", "aside"]
    for elemento in soup(etiquetas_basura):
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
    Convierte un CSV a texto de forma ultra-resistente.
    """
    texto = ""
    try:
        # Intento 1: Pandas detectando el separador automáticamente
        df = pd.read_csv(
            ruta, 
            sep=None,          # Detecta automáticamente si son comas o punto y coma
            engine='python',   # Fuerza el uso de Python
            on_bad_lines='skip',
            encoding_errors='ignore'
        )
        for _, fila in df.iterrows():
            for columna in df.columns:
                texto += f"{columna}: {str(fila[columna])}\n"
            texto += "\n"

    except Exception:
        # Intento 2: Si Pandas falla, usamos el lector nativo (A prueba de balas)
        import csv
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            lector = csv.reader(f)
            try:
                encabezados = next(lector) # Leer la primera fila (nombres de columnas)
            except StopIteration:
                return "" # El archivo estaba vacío
            
            for fila in lector:
                for i in range(len(fila)):
                    if i < len(encabezados):
                        texto += f"{encabezados[i]}: {fila[i]}\n"
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
        print(f"Procesando: {os.path.basename(ruta_archivo)}")
        texto = extraer_texto(ruta_archivo)
        texto = limpiar_texto(texto)
        idioma = identificar_idioma(texto)
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

"""def procesar_carpeta(carpeta, fenomeno):
    documentos = []
    formatos = (
        ".pdf", ".html", ".htm", ".json", ".csv", ".xlsx", 
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pbf"
    )
    contador = 1

    for raiz, directorios, archivos in os.walk(carpeta):
        for archivo in sorted(archivos):
            if archivo.lower().endswith(formatos):
                
                ruta = os.path.join(raiz, archivo)
                ruta = os.path.abspath(ruta)
    
                if os.name == 'nt':  # Si usas Windows, rompe el límite de 260 caracteres
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
"""
# ==========================================================
# PROCESAR TODOS LOS DOCUMENTOS DE UNA CARPETA
# ==========================================================

def procesar_carpeta(carpeta, fenomeno):
    """
    Procesa todos los archivos soportados de una carpeta y sus subcarpetas.
    """
    documentos = []
    formatos = (
        ".pdf", ".html", ".htm", ".json", ".csv", ".xlsx", 
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pbf"
    )
    contador = 1

    for raiz, directorios, archivos in os.walk(carpeta):
        for archivo in sorted(archivos):
            if archivo.lower().endswith(formatos):
                
                ruta = os.path.join(raiz, archivo)
                ruta = os.path.abspath(ruta)
    
                if os.name == 'nt':  # Si usas Windows, rompe el límite de 260 caracteres
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
    # 1. Rutas de entrada y salida
    # (AQUÍ PUEDES CAMBIAR LA CARPETA SI HACES LA PRUEBA)
    carpeta = "datos/corpus"
    archivo_salida = "datos/texto_extraido.json"
    fenomeno = 1

    if not os.path.exists(carpeta):
        print(f"La carpeta '{carpeta}' no existe.")
        print("Créala dentro del proyecto y coloca allí los archivos.")
    else:
        print("Iniciando extracción. Esto puede tardar varios minutos dependiendo de los PDFs...")
        
        # 2. Ejecutar la extracción
        documentos = procesar_carpeta(carpeta, fenomeno)

        print("\n")
        print("=" * 70)
        print(f"Se procesaron {len(documentos)} documento(s).")
        print("=" * 70)

        # 3. Guardar todo en un archivo JSON
        print(f"\nGuardando resultados en: {archivo_salida}...")
        
        # Guardamos el archivo asegurando que los acentos y las eñes se vean bien (UTF-8)
        with open(archivo_salida, "w", encoding="utf-8") as archivo_json:
            json.dump(documentos, archivo_json, ensure_ascii=False, indent=4)
            
        print("¡Misión del Extractor completada con éxito! Archivo listo para el Chunker.")
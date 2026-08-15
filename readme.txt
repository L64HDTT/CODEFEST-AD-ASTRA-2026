**INSTRUCCIONES __PREVIAS__ A CORRER EL CÓDIGO:**
1. Descargar tesseract-OCR y añadirlo al PATH. --- >> PATH recomendado --->> C:\Program Files\Tesseract-OCR\

2. Instalar librerias y dependencias directamente en la terminal usando el requirements.txt:
    **--->> pip install -r requirements.txt <<---**

**INSTRUCCIONES PARA CORRER EL CÓDIGO:**
El proceso principal se realiza en dos partes diferentes. La primera mitad consiste en la extracción y embedding de los textos,
la segunda parte consiste en la generación de texto para las consultas.
    Primer paso:
        - Correr 'indexador.py'
    Segundo paso:
        - Correr 'generador.py'

Esencialmente, esto permite la divisón de los procesos. Permitiendo la generación de más consultas sin tener que volver a leer
todo el corpus. 


Para realizar la primera parte del proceso se debe correr una única vez el script 'indexador.py'. La segunda parte solo require
de correr el script 'generador.py'

Para la construcción del grafo solo se debe correr el script 'constructor_grafo.py'
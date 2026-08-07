import os
import re
import json
import fitz
import pandas as pd

from bs4 import BeautifulSoup

from PIL import Image

import pytesseract

from langdetect import detect, DetectorFactory
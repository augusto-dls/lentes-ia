import os
import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import anthropic
from google import genai as google_genai
from google.genai import types
from openai import OpenAI
from gtts import gTTS
import tempfile

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------- CONFIGURACIÓN DE CLIENTES ----------

# ---------- CONFIGURACIÓN DE CLIENTES ----------

anthropic_client = None
openai_client = None

if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    logger.info("Claude disponible")

if os.getenv("GEMINI_API_KEY"):
    #genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("Gemini disponible")

if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("GPT-4o disponible")

# ---------- PROMPT ----------

PROMPT = """Sos un asistente de descripción visual para personas con discapacidad visual.
Analizá la imagen y respondé en dos partes, de forma clara y concisa,
en el mismo idioma que aparezca en la imagen o en español si no hay texto.

PARTE 1 - DESCRIPCIÓN:
En 1 o 2 oraciones describí lo esencial: qué objeto o escena se ve,
su contexto inmediato y cualquier detalle relevante para identificarlo
o interactuar con él (colores, estado, posición, cantidad sólo si aplica).
No describas lo obvio ni lo decorativo. Priorizá información útil y accionable.

PARTE 2 - TEXTO VISIBLE:
Transcribí literalmente todo el texto que aparezca en la imagen
(etiquetas, carteles, pantallas, envases, documentos, etc).
Si hay múltiples bloques de texto, separarlos con un guión.
Si no hay texto, respondé únicamente: Sin texto visible.

Respondé SIEMPRE en este formato:
Descripción: [texto]
Texto visible: [texto]"""

# ---------- FUNCIONES POR MODELO ----------

def analizar_con_claude(imagen_base64: str, tipo: str) -> str:
    logger.info("Intentando con Claude...")
    mensaje = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": tipo,
                        "data": imagen_base64,
                    },
                },
                {"type": "text", "text": PROMPT}
            ],
        }]
    )
    return mensaje.content[0].text


def analizar_con_gemini(imagen_bytes: bytes, tipo: str) -> str:
    logger.info("Intentando con Gemini...")
    cliente = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    respuesta = cliente.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=imagen_bytes, mime_type=tipo),
            PROMPT
        ]
    )
    return respuesta.text


def analizar_con_gpt4o(imagen_base64: str, tipo: str) -> str:
    logger.info("Intentando con GPT-4o...")
    respuesta = openai_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{tipo};base64,{imagen_base64}"
                    }
                },
                {"type": "text", "text": PROMPT}
            ]
        }]
    )
    return respuesta.choices[0].message.content


# ---------- FALLBACK AUTOMÁTICO ----------

def analizar_imagen_con_fallback(imagen_bytes: bytes, imagen_base64: str, tipo: str) -> str:
    modelos = []

    if anthropic_client:
        modelos.append(("Claude", lambda: analizar_con_claude(imagen_base64, tipo)))
    if os.getenv("GEMINI_API_KEY"):
        modelos.append(("Gemini", lambda: analizar_con_gemini(imagen_bytes, tipo)))
    if openai_client:
        modelos.append(("GPT-4o", lambda: analizar_con_gpt4o(imagen_base64, tipo)))

    if not modelos:
        raise HTTPException(status_code=503, detail="No hay API keys configuradas.")

    errores = []
    for nombre, funcion in modelos:
        try:
            resultado = funcion()
            logger.info(f"Respuesta obtenida con {nombre}")
            return resultado
        except Exception as e:
            logger.warning(f"{nombre} falló: {e}")
            errores.append(f"{nombre}: {e}")

    raise HTTPException(
        status_code=503,
        detail=f"Todos los modelos fallaron. Errores: {' | '.join(errores)}"
    )

# ---------- ENDPOINT ----------

@app.post("/analizar")
async def analizar(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    imagen_base64 = base64.standard_b64encode(contenido).decode("utf-8")
    tipo = imagen.content_type or "image/jpeg"

    texto = analizar_imagen_con_fallback(contenido, imagen_base64, tipo)

    tts = gTTS(text=texto, lang="es")
    archivo_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(archivo_temp.name)

    return FileResponse(
        path=archivo_temp.name,
        media_type="audio/mpeg",
        filename="respuesta.mp3"
    )


@app.get("/")
def root():
    return {"estado": "servidor activo"}

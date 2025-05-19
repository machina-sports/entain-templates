from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from openai import OpenAI

import base64


def generate_image(request_data):

    headers = request_data.get("headers")

    params = request_data.get("params")

    api_key = headers.get("api_key", "")

    image_id = params.get("image_id", "")

    model_name = params.get("model", "")

    instruction = params.get("instruction", "")

    if not api_key:
        return {"status": "error", "message": "API key is required."}

    if not model_name:
        return {"status": "error", "message": "Model name is required."}

    prompt = f"""
        Create a 3:2 landscape image blog thumbnail, strictly aligned with the imagery guidelines below, using the provided <brandGuidelines> XML:

        Imagery Guidelines:

        Foreground Illustration Style - applies to any foreground subjects supplied in {instruction}:
        - Illustrate each foreground subject in a style reminiscent of the classic SNES video game "International Superstar Soccer" – true 16-bit palette, visible 2–3 px outlines, pixel-inspired aesthetics, and dynamic, exaggerated poses typical of retro sports games.  
        - Jerseys or clothing must clearly match each team’s official colors while omitting sponsor logos or recognizable faces.  
        - Convey energetic movement through pixel-art action lines and diagonal streaks; avoid motion blur to preserve crisp retro authenticity.

        General Design Style (all other elements):
        Maintain sharp, modern, professional treatments for backgrounds, DOT graphic elements, and overall composition. Lighting should be high-contrast and crisp, simulating stadium floodlights without HDR glow.

        DOT Graphic Element Usage:
        MASK – Crop the entire foreground illustration within a 900 × 700 px parallelogram tilted 12° right, corner radius 65 px.  
        FILLED – Place a solid DOT in Bright Blue (#0A5EEA) or Bright Deep Blue (#003DC4) behind the foreground subjects to add visual depth.

        Background Canvas Options:
        - Bright Deep Blue (#003DC4) or Bright Dark Blue (#061F3F) for a dramatic night-game mood.  
        - White (#FFFFFF) for a lighter, open feel.

        Gradient (optional, use sparingly):
        – Bright Extra Light Blue (#D3ECFF) to Bright Light Blue (#45CAFF): soft diagonal sweep.  
        – Bright Light Blue (#45CAFF) to Bright Blue (#0A5EEA): energetic diagonal transition.  
        – Bright Deep Blue (#003DC4) to Bright Dark Blue (#061F3F): vertical fade for night intensity.

        Composition:
        Reserve 15–20 percent negative space for text overlays (headlines, match details).

        Camera POV:
        Low-angle or sideline telephoto with shallow depth of field (ƒ2.8–ƒ4).

        Lighting:
        High contrast from stadium floodlights, crisp shadows, no HDR glow.

        Mandatory Restrictions:
        - No sponsorship logos on jerseys.  
        - Never include the Sportingbet logo, word-mark, or symbol.


        Full Integrated Brand Guidelines XML:

        <brandGuidelines>
        <brandName>Sportingbet</brandName>

        <colors>
            <marketingColors>
            <color name="Bright Blue" hex="#0A5EEA" prompt="Use for large, vivid background panels or oversized DOT fills to signal energy."/>
            <color name="Bright Deep Blue" hex="#003DC4" prompt="Primary solid backdrop on dark layouts; pairs with white or bright light blue foreground elements."/>
            <color name="Bright Dark Blue" hex="#061F3F" prompt="Deep navy shadow tone—excellent for contrast behind cyan accents or white text."/>
            <color name="Bright Extra Light Blue" hex="#D3ECFF" prompt="Soft cyan wash for subtle sections, gradient starts, or secondary panels."/>
            <color name="Bright Light Blue" hex="#45CAFF" prompt="Electric pop for gradient end-points, call-to-action strokes, or accent motion trails."/>
            <color name="Bright Red" hex="#F13131" prompt="High-impact outline of the DOT symbol or dynamic swoosh lines that guide the eye."/>
            <color name="White" hex="#FFFFFF" prompt="Neutral void to let vibrant blues breathe—ideal for negative space and clean type."/>
            <color name="Light Grey" hex="#EEEFF1" prompt="Background canvas for accessibility charts or UI components that must stay subdued."/>
            </marketingColors>
        </colors>

        <graphicElements>
            <dotSymbol prompt="Create a 900×700-pixel rounded parallelogram; tilt it 12° to the right; apply a 65-pixel corner radius. It can act as an image mask or a bold filled block.">
            <dimensions width="900px" height="700px"/>
            <tilt direction="right" degrees="12"/>
            <cornerRadius pixels="65px"/>
            <usage>
                <imageContainer>true</imageContainer>
                <outline>false</outline>
                <filledShape>true</filledShape>
            </usage>
            </dotSymbol>

        <linearGradients>
        <gradient startColor="Bright Extra Light Blue" endColor="Bright Light Blue"/>
        <gradient startColor="Bright Light Blue" endColor="Bright Blue"/>
        <gradient startColor="Bright Deep Blue" endColor="Bright Dark Blue"/>
        <gradientPrompts>
            <prompt>Soft sky-to-electric sweep (top-left ➜ bottom-right) for uplifting tech vibe.</prompt>
            <prompt>Medium cyan into royal blue diagonal for energetic, modern sports feel.</prompt>
            <prompt>Deep ocean fade to midnight navy vertical for dramatic, night-game intensity.</prompt>
        </gradientPrompts>
        <guidelines>
            <guideline>Use sparingly to complement design, not overpower it.</guideline>
            <guideline>Apply only to large areas, never to text or small elements.</guideline>
            <guideline>Ensure smooth transitions without harsh steps.</guideline>
        </guidelines>
        </linearGradients>

        </graphicElements>

        <mandatoryGuidelines>
            <noSponsorLogos>true</noSponsorLogos>
            <noSportingbetLogo>true</noSportingbetLogo>
        </mandatoryGuidelines>

        </brandGuidelines>

    """

    try:
        llm = OpenAI(api_key=api_key)

        result = llm.images.generate(
            model=model_name,
            prompt=prompt,
            size="1536x1024",
            quality="high",
        )

        image_base64 = result.data[0].b64_json

        image_bytes = base64.b64decode(image_base64)

        full_filepath = f"/work/images/{image_id}.webp"

        with open(full_filepath, 'wb') as f:

            f.write(image_bytes)

        final_filename = f"{image_id}.webp"

        result = {
            "final_filename": final_filename,
            "full_filepath": full_filepath
        }

        return {"status": True, "data": result, "message": "Image generated."}

    except Exception as e:
        return {"status": False, "message": f"Exception when generating image: {e}"} 
    

def invoke_embedding(params):

    api_key = params.get("api_key", "")

    model_name = params.get("model_name")

    if not api_key:
        return {"status": "error", "message": "API key is required."}

    if not model_name:
        return {"status": "error", "message": "Model name is required."}

    try:
        llm = OpenAIEmbeddings(api_key=api_key, model=model_name)
        # llm = OpenAI(api_key=api_key)

    except Exception as e:
        return {"status": "error", "message": f"Exception when creating model: {e}"}

    return {"status": True, "data": llm, "message": "Model loaded."}


def invoke_prompt(params):

    api_key = params.get("api_key")

    model_name = params.get("model_name")

    if not api_key:
        return {"status": "error", "message": "API key is required."}

    if not model_name:
        return {"status": "error", "message": "Model name is required."}

    try:
        llm = ChatOpenAI(model=model_name, api_key=api_key)

    except Exception as e:
        return {"status": "error", "message": f"Exception when creating model: {e}"}

    return {"status": True, "data": llm, "message": "Model loaded."}


def transcribe_audio_to_text(params):
    """
    Transcribe an audio file to text using the new OpenAI Whisper transcription API.

    :param params: Dictionary containing the 'api_key' and 'audio-path' parameters.
    :return: Transcribed text or error message.
    """

    api_key = params.get("headers").get("api_key")
    file_items = params.get("params").get("audio-path", [])

    audio_file_path = file_items[0]

    try:

        llm = OpenAI(api_key=api_key)

        with open(audio_file_path, "rb") as audio_file:
            print(f"Transcribing file: {audio_file_path}")

            transcript = llm.audio.transcriptions.create(
              model="whisper-1",
              file=audio_file
            )

        return {"status": True, "data": transcript.text}

    except Exception as e:
        return {"status": False, "message": f"Exception when transcribing audio: {e}"} 

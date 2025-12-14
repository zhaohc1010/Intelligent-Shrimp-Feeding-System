# Filename: app.py
# Description: Main Application - Handles AI, Interface, and Command Dispatch

import os
import cv2
import base64
import time
import asyncio
import aiohttp
import json
import struct
import gzip
import uuid
import logging
import tempfile
import threading
import queue
import sys
import io
import re  # Used for command parsing

# *** (Core Modification) ***
# Import RelayMotorController class from local file
from relay_motor_controller import RelayMotorController

# Basic Libraries
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import numpy as np
from pynput import keyboard
from openai import OpenAI
import requests
from pydub import AudioSegment
from pydub.playback import play

# --- Global Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Load configuration from config.py ---
try:
    from config import vision_model_config, asr_model_config, tts_model_config
except ImportError:
    logger.error("Error: Configuration file 'config.py' not found or incomplete.")
    logger.error("Please ensure 'config.py' exists.")
    sys.exit(1)


# ==============================================================================
#  (New) Command Parsing and Dispatch (v2 - Extended Synonyms)
# ==============================================================================
def parse_command(prompt: str, motor_ctrl: RelayMotorController) -> bool:
    """
    Parses voice commands.
    - If it is a motor command, execute it and return True.
    - If not, return False.
    """
    prompt = prompt.strip().lower()  # Convert to lowercase for English matching

    # --- (New) Synonym Lists (English) ---
    # Action: Up
    UP_KEYWORDS = ["up", "raise", "lift", "ascend", "pull up", "go up"]
    # Action: Down
    DOWN_KEYWORDS = ["down", "lower", "drop", "descend", "go down"]
    # Action: Power On
    ON_KEYWORDS = ["power on", "turn on", "motor on", "start motor", "open motor"]
    # Action: Power Off
    OFF_KEYWORDS = ["power off", "turn off", "motor off", "stop motor", "close motor"]

    # Find numbers (seconds) in the command
    match = re.search(r'(\d+)', prompt)  # Match 1 or more digits
    duration = None
    if match:
        try:
            duration = int(match.group(1))
        except:
            pass  # Ignore if number parsing fails

    # --- Check for Action Commands with Duration ---
    if duration:
        # Check for any "Up" synonyms
        if any(keyword in prompt for keyword in UP_KEYWORDS):
            logger.info(f"Executing Motor Command: UP for {duration} seconds")
            text_to_speech(f"OK, moving platform up for {duration} seconds")
            motor_ctrl.move_up(duration)
            return True

        # Check for any "Down" synonyms
        if any(keyword in prompt for keyword in DOWN_KEYWORDS):
            logger.info(f"Executing Motor Command: DOWN for {duration} seconds")
            text_to_speech(f"OK, moving platform down for {duration} seconds")
            motor_ctrl.move_down(duration)
            return True

    # --- Check for On/Off Commands (No Duration) ---
    if any(keyword in prompt for keyword in ON_KEYWORDS):
        logger.info("Executing Motor Command: Power ON")
        text_to_speech("OK, motor power turned on")
        motor_ctrl.open_motor()
        return True

    if any(keyword in prompt for keyword in OFF_KEYWORDS):
        logger.info("Executing Motor Command: Power OFF")
        text_to_speech("OK, motor power turned off")
        motor_ctrl.close_motor()
        return True

    # --- Non-Motor Command ---
    # If we reach here, no keywords matched. It is not a motor command.
    return False


# ==============================================================================
#  Text-to-Speech (TTS) Section
# ==============================================================================
def text_to_speech(text_to_speak: str):
    appID = tts_model_config.get("app_id")
    accessKey = tts_model_config.get("access_key")
    resourceID = tts_model_config.get("resource_id")

    if not all([appID, accessKey, resourceID]):
        logger.error("Error: TTS configuration incomplete. Please check config.py.")
        return

    url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    headers = {
        "X-Api-App-Id": appID,
        "X-Api-Access-Key": accessKey,
        "X-Api-Resource-Id": resourceID,
        "Content-Type": "application/json",
        "Connection": "keep-alive"
    }

    # Note: 'speaker' is set to a Chinese voice ID.
    # Ideally, change this to an English voice ID if available (e.g., 'bv406_streaming').
    # Currently keeping original ID but changing explicit_language to 'en'.
    payload = {
        "user": {"uid": "123123"},
        "req_params": {
            "text": text_to_speak,
            "speaker": "zh_male_shaonianzixin_moon_bigtts",
            "audio_params": {"format": "mp3", "sample_rate": 24000, "enable_timestamp": False, "speech_rate": 50},
            "additions": json.dumps({"explicit_language": "en"})  # Changed to English
        }
    }

    logger.info(f"Synthesizing speech: \"{text_to_speak}\"")
    session = requests.Session()
    response = None
    try:
        response = session.post(url, headers=headers, json=payload, stream=True)
        if response.status_code != 200:
            logger.error(f"TTS Request Failed: {response.status_code} {response.text}")
            return
        audio_data = bytearray()
        for chunk in response.iter_lines(decode_unicode=True):
            if not chunk: continue
            data = json.loads(chunk)
            if data.get("code", 0) == 0 and "data" in data and data["data"]:
                audio_data.extend(base64.b64decode(data["data"]))
            elif data.get("code", 0) > 0 and data.get("code", 0) != 20000000:
                logger.error(f"TTS Service Error: {data}")
                break
        if audio_data:
            logger.info("Speech synthesis complete, playing...")
            audio_stream = io.BytesIO(audio_data)
            sound = AudioSegment.from_file(audio_stream, format="mp3")
            play(sound)
            logger.info("Playback finished.")
        else:
            logger.warning("TTS returned no valid audio data.")
    except Exception as e:
        logger.error(f"Error during TTS synthesis or playback: {e}")
    finally:
        if response: response.close()
        session.close()


# ==============================================================================
#  Vision Model Section
# ==============================================================================
def analyze_frames_with_vision_model(client, model_endpoint, captured_frames, user_prompt):
    if not captured_frames:
        logger.error("Error: Failed to capture frames, cannot proceed with analysis.")
        text_to_speech("Sorry, I didn't see clearly, please try again.")
        return

    logger.info("\nFrame capture complete, sending to model for analysis...")
    logger.info(f"Command sent: \"{user_prompt}\"")

    try:
        message_content = []
        for url in captured_frames:
            message_content.append({"type": "image_url", "image_url": {"url": url}})
        message_content.append({"type": "text", "text": user_prompt})

        response = client.chat.completions.create(
            model=model_endpoint,
            messages=[{"role": "user", "content": message_content}],
        )
        if response.choices:
            analysis_result = response.choices[0].message.content
            logger.info(f"\n--- Model Analysis Result --- \n{analysis_result}\n---------------------\n")
            text_to_speech(analysis_result)
        else:
            logger.warning("API call successful but returned no valid result.")
            text_to_speech("The model returned no results, please try again later.")
    except Exception as e:
        logger.error(f"Error calling Vision API: {e}")
        text_to_speech("Visual analysis error, please check network or API settings.")


def process_visual_analysis_task(cap, client, endpoint, prompt, duration, num_frames):
    logger.info(f"Capturing {num_frames} frames over {duration} seconds.")
    captured_frames = []
    start_time = time.time()
    while time.time() - start_time < duration:
        ret_cap, frame_cap = cap.read()
        if ret_cap:
            _, buffer = cv2.imencode('.jpg', frame_cap)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{base64_image}"
            captured_frames.append(image_data_url)
        time.sleep(duration / num_frames)
    analyze_frames_with_vision_model(client, endpoint, captured_frames, prompt)
    logger.info("Analysis complete! Returned to live preview mode, awaiting next command.")


# ==============================================================================
#  Vision Main Loop (Includes Command Dispatch Logic)
# ==============================================================================
def vision_main_loop(result_queue: queue.Queue, motor_controller: RelayMotorController):
    """
    Main function handling camera display and background analysis tasks.
    """
    vision_api_key = vision_model_config.get('api_key')
    vision_model_endpoint = vision_model_config.get('model_endpoint')
    num_frames_to_capture = 60
    capture_duration_seconds = 3

    if not vision_api_key or "replace here" in vision_api_key:
        logger.error("Error: 'api_key' not set in config.py.")
        return

    try:
        vision_client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=vision_api_key,
        )
    except Exception as e:
        logger.error(f"Error initializing vision model client: {e}")
        return

    cap = cv2.VideoCapture(0)  # Use local camera
    if not cap.isOpened():
        logger.error("Error: Cannot open camera.")
        return

    window_name = "AI Vision Assistant (Hold Space to Speak | Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 500)
    logger.info("System Ready. Hold [Spacebar] to speak, release to analyze.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Cannot read camera frame.")
                break
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                logger.info("User exited program.")
                break

            try:
                user_prompt = result_queue.get_nowait()
                if user_prompt:
                    logger.info(f"\nReceived Voice Command: \"{user_prompt}\"")

                    # *** Core Command Dispatch Logic ***
                    is_motor_command = parse_command(user_prompt, motor_controller)

                    if not is_motor_command:
                        # Not a motor command, execute visual analysis
                        logger.info("Non-motor command detected, starting visual analysis...")
                        wait_message = f"OK, please wait, I will analyze the next {capture_duration_seconds} seconds of video."
                        logger.info(wait_message)

                        tts_thread = threading.Thread(target=text_to_speech, args=(wait_message,), daemon=True)
                        tts_thread.start()

                        analysis_thread = threading.Thread(
                            target=process_visual_analysis_task,
                            args=(cap, vision_client, vision_model_endpoint, user_prompt, capture_duration_seconds,
                                  num_frames_to_capture),
                            daemon=True
                        )
                        analysis_thread.start()
                    else:
                        # Motor command handled by parse_command
                        logger.info("Motor command executed, returning to standby.")

            except queue.Empty:
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera closed.")


# ==============================================================================
#  Speech Recognition (ASR) Section
# ==============================================================================

DEFAULT_SAMPLE_RATE = 16000
CHANNELS = 1


class ASRConfig:
    def __init__(self):
        self.auth = {"app_key": asr_model_config.get("app_key"), "access_key": asr_model_config.get("access_key")}

    @property
    def app_key(self) -> str: return self.auth["app_key"]

    @property
    def access_key(self) -> str: return self.auth["access_key"]


asr_config = ASRConfig()


class ProtocolVersion: V1 = 0b0001


class MessageType:
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ERROR_RESPONSE = 0b1111


class MessageTypeSpecificFlags:
    NO_SEQUENCE, POS_SEQUENCE, NEG_SEQUENCE, NEG_WITH_SEQUENCE = 0b0000, 0b0001, 0b0010, 0b0011


class SerializationType: NO_SERIALIZATION, JSON = 0b0000, 0b0001


class CompressionType: GZIP = 0b0001


class CommonUtils:
    @staticmethod
    def gzip_compress(data: bytes) -> bytes:
        return gzip.compress(data)

    @staticmethod
    def gzip_decompress(data: bytes) -> bytes:
        return gzip.decompress(data)

    @staticmethod
    def read_wav_info(data: bytes) -> tuple:
        if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE': raise ValueError("Invalid WAV file")
        num_channels = struct.unpack('<H', data[22:24])[0]
        sample_rate = struct.unpack('<I', data[24:28])[0]
        bits_per_sample = struct.unpack('<H', data[34:36])[0]
        pos = 36
        while pos < len(data) - 8:
            subchunk_id, subchunk_size = data[pos:pos + 4], struct.unpack('<I', data[pos + 4:pos + 8])[0]
            if subchunk_id == b'data':
                return (num_channels, bits_per_sample // 8, sample_rate,
                        subchunk_size // (num_channels * (bits_per_sample // 8)), data[pos + 8:pos + 8 + subchunk_size])
            pos += 8 + subchunk_size
        raise ValueError("Invalid WAV file: no data subchunk found")


class AsrRequestHeader:
    def __init__(self):
        self.message_type = MessageType.CLIENT_FULL_REQUEST
        self.message_type_specific_flags = MessageTypeSpecificFlags.POS_SEQUENCE
        self.serialization_type = SerializationType.JSON
        self.compression_type = CompressionType.GZIP
        self.reserved_data = bytes([0x00])

    def to_bytes(self) -> bytes:
        return bytes([(ProtocolVersion.V1 << 4) | 1, (self.message_type << 4) | self.message_type_specific_flags,
                      (self.serialization_type << 4) | self.compression_type]) + self.reserved_data

    def with_message_type(self, mt: int): self.message_type = mt; return self

    def with_message_type_specific_flags(self, flags: int): self.message_type_specific_flags = flags; return self

    @staticmethod
    def default_header(): return AsrRequestHeader()


class RequestBuilder:
    @staticmethod
    def new_auth_headers() -> dict:
        return {"X-Api-Resource-Id": "volc.bigasr.sauc.duration", "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Access-Key": asr_config.access_key, "X-Api-App-Key": asr_config.app_key}

    @staticmethod
    def new_full_client_request(seq: int) -> bytes:
        header = AsrRequestHeader.default_header()
        payload = {"user": {"uid": "demo_uid"},
                   "audio": {"format": "wav", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
                   "request": {"model_name": "bigmodel", "enable_itn": True, "enable_punc": True,
                               "show_utterances": True}}
        compressed_payload = CommonUtils.gzip_compress(json.dumps(payload).encode('utf-8'))
        return header.to_bytes() + struct.pack('>i', seq) + struct.pack('>I',
                                                                        len(compressed_payload)) + compressed_payload

    @staticmethod
    def new_audio_only_request(seq: int, segment: bytes, is_last: bool = False) -> bytes:
        header = AsrRequestHeader.default_header().with_message_type(MessageType.CLIENT_AUDIO_ONLY_REQUEST)
        if is_last: header.with_message_type_specific_flags(MessageTypeSpecificFlags.NEG_WITH_SEQUENCE); seq = -seq
        compressed_segment = CommonUtils.gzip_compress(segment)
        return header.to_bytes() + struct.pack('>i', seq) + struct.pack('>I',
                                                                        len(compressed_segment)) + compressed_segment


class ResponseParser:
    @staticmethod
    def parse_response(msg: bytes) -> dict:
        header_size = msg[0] & 0x0f;
        payload = msg[header_size * 4:]
        message_type_specific_flags = msg[1] & 0x0f
        is_last_package = bool(message_type_specific_flags & 0x02)
        payload_msg = None
        if message_type_specific_flags & 0x01: payload = payload[4:]
        if message_type_specific_flags & 0x04: payload = payload[4:]
        if msg[1] >> 4 == MessageType.SERVER_FULL_RESPONSE:
            payload = payload[4:]
        elif msg[1] >> 4 == MessageType.SERVER_ERROR_RESPONSE:
            payload = payload[8:]
        if payload:
            if msg[2] & 0x0f == CompressionType.GZIP: payload = CommonUtils.gzip_decompress(payload)
            if msg[2] >> 4 == SerializationType.JSON: payload_msg = json.loads(payload.decode('utf-8'))
        return {"is_last_package": is_last_package, "payload_msg": payload_msg}


class AsrWsClient:
    def __init__(self, url: str, segment_duration: int = 200):
        self.seq = 1;
        self.url = url;
        self.segment_duration = segment_duration

    async def execute(self, file_path: str):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.url, headers=RequestBuilder.new_auth_headers()) as conn:
                with open(file_path, 'rb') as f:
                    content = f.read()
                segment_size = CommonUtils.read_wav_info(content)[2] * self.segment_duration // 1000 * CHANNELS * 2
                await conn.send_bytes(RequestBuilder.new_full_client_request(self.seq));
                self.seq += 1
                await conn.receive()
                audio_segments = [content[i:i + segment_size] for i in range(0, len(content), segment_size)]
                for i, segment in enumerate(audio_segments):
                    await conn.send_bytes(RequestBuilder.new_audio_only_request(self.seq, segment,
                                                                                is_last=(i == len(audio_segments) - 1)))
                    if not (i == len(audio_segments) - 1): self.seq += 1
                    await asyncio.sleep(self.segment_duration / 1000)
                async for msg in conn:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        response = ResponseParser.parse_response(msg.data)
                        yield response
                        if response["is_last_package"]: break


class AudioRecorder:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, channels=CHANNELS):
        self.samplerate, self.channels, self.recording, self.frames = samplerate, channels, False, []

    def start(self):
        self.frames = [];
        self.recording = True;
        logger.info("Recording started...")

    def stop(self) -> str:
        self.recording = False;
        logger.info("Recording finished, processing...")
        if not self.frames: logger.warning("No audio recorded."); return None
        recording_data = np.concatenate(self.frames, axis=0)
        recording_data_int16 = (recording_data * 32767).astype(np.int16)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
                filepath = temp_file.name
                write_wav(temp_file, self.samplerate, recording_data_int16)
            logger.info(f"Recording saved to temp file: {filepath}");
            return filepath
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}");
            return None

    def callback(self, indata, frames, time, status):
        if self.recording: self.frames.append(indata.copy())


async def process_audio_file(filepath: str) -> str:
    url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
    accumulated_text = ""
    logger.info("Connecting to Speech Recognition service and sending audio...")
    try:
        async for response in AsrWsClient(url).execute(filepath):
            payload = response.get("payload_msg", {})
            if payload and 'result' in payload:
                result_data = payload['result']
                if isinstance(result_data, dict) and 'text' in result_data:
                    text_from_result = result_data.get('text', '')
                    if text_from_result: accumulated_text = text_from_result
                elif isinstance(result_data, list):
                    current_message_text = ""
                    for utt in result_data:
                        if isinstance(utt, dict) and 'text' in utt: current_message_text += utt.get('text', '')
                    if current_message_text: accumulated_text = current_message_text
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
    finally:
        if os.path.exists(filepath): os.remove(filepath)
    if accumulated_text:
        logger.info(f"Speech Recognition Complete: {accumulated_text}")
    else:
        logger.warning("Speech Recognition returned no valid text.")
    return accumulated_text


async def asr_main(result_queue: queue.Queue):
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()
    recorder = AudioRecorder()

    def on_press(key):
        if key == keyboard.Key.space and not recorder.recording: recorder.start()

    def on_release(key):
        if key == keyboard.Key.space and recorder.recording:
            filepath = recorder.stop()
            if filepath: loop.call_soon_threadsafe(audio_queue.put_nowait, filepath)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    with sd.InputStream(samplerate=recorder.samplerate, channels=recorder.channels, callback=recorder.callback):
        while True:
            filepath_to_process = await audio_queue.get()
            recognized_text = await process_audio_file(filepath_to_process)
            if recognized_text:
                result_queue.put(recognized_text)


def run_asr_thread(result_queue: queue.Queue):
    if not asr_config.app_key or "replace here" in asr_config.app_key:
        logger.error("Error: app_key/access_key not set in config.py. Voice features disabled.")
        return
    asyncio.run(asr_main(result_queue))


# ==============================================================================
#  Main Program Entry Point
# ==============================================================================
if __name__ == "__main__":

    # 1. Initialize Motor Controller
    # !!! Note: Check if 'port' matches your relay's serial port (e.g., COM3, /dev/ttyUSB0)
    motor_controller = RelayMotorController(port='COM3')

    # 2. Initialize Voice Command Queue
    speech_result_queue = queue.Queue()

    # 3. Use try...finally to ensure relay is safely closed
    try:
        # Run Speech Recognition in background thread
        asr_thread = threading.Thread(target=run_asr_thread, args=(speech_result_queue,), daemon=True)
        asr_thread.start()

        # Run Camera and Main Logic in main thread
        vision_main_loop(speech_result_queue, motor_controller)

    finally:
        # 4. Cleanup on exit
        logger.info("Program exiting, cleaning up...")
        motor_controller.cleanup()
        logger.info("Cleanup done.")
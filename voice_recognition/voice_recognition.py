import speech_recognition as sr
import threading
import queue

command_queue = queue.Queue()

def listen_for_commands():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            try:
                audio = r.listen(source, timeout=5)
                command = r.recognize_google(audio).lower()
                command_queue.put(command)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                # 네트워크 문제 발생시 잠시 대기 후 재시도
                time.sleep(5)

def start_voice_recognition():
    thread = threading.Thread(target=listen_for_commands)
    thread.daemon = True
    thread.start()

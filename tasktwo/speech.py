import pyttsx3 as tts
def main(text):
    speech=tts.init()
    answer="The focal length is "+str(text)+"mm"
    speech.say(answer)
    speech.runAndWait()
if __name__=='__main__':
    def __init__(self,value):
        main(value)
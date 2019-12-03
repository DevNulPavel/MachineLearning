#!/usr/bin/env python3

import speech_recognition

class Statement:
    def __init__(self, dict):
        self.confidence = dict["confidence"]
        self.text = dict["transcript"].lower()

    def __repr__(self):
        return "[{}] {}".format(self.confidence, self.text)

    def __str__(self):
        return self.text

    def __gt__(self, other):
        return self.confidence > other.confidence

def check_in_string(string, words):
    if any(word in string for word in words):
        return True
    return False

def processStatement(best_statement, statements):
    threshold = 0.4
    if best_statement is None or best_statement.confidence < threshold:
        answer = "Простите, вас плохо слышно"
    else:
        # Some examples of commands
        # Check all received statements (even with smaller confidence)
        # Because we need more confidence in command will be recognized the first time
        command_recognized = False
        be_quiet = False
        for st in statements:
            if check_in_string(st.text, ('вперёд', 'иди', 'шагай')):
                answer = "Я знаю эту команду!"
                command_recognized = True
            elif check_in_string(st.text, ('остановись', 'стоп', 'стой')):
                answer = "Я знаю эту команду!"
                command_recognized = True
            elif check_in_string(st.text, ('тихо', 'молчать', 'тишина', 'тише')):
                be_quiet = True
                command_recognized = True
                answer = "Я буду вести себя тише"
            elif check_in_string(st.text, ('говори', 'громче')):
                be_quiet = False
                answer = "Я буду говорить громче"
                command_recognized = True
        if not command_recognized:
            answer = make_answer(best_statement.text)  # takes many time to be executed
    return answer

def chooseBestStatement(statements):
    if statements:
        return max(statements, key=lambda s: s.confidence)
    else:
        return None

def jsonToStatements(json):
    threshold = 0.4
    statements = []
    if len(json) is not 0:
        for info in json["alternative"]:
            if "confidence" not in info:
                info["confidence"] = threshold + 0.1
            newStatement = Statement(info)
            statements.append(newStatement)
    return statements


def recognizeWithGoogle(r, audio):
    try:
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        json = r.recognize_google(audio, language="ru_RU", show_all=True)

        statements = jsonToStatements(json)
        print("Google: ", statements)

        bestStatement = chooseBestStatement(statements)

        processStatement(best_statement, statements)

    except speech_recognition.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except speech_recognition.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def main():
    r = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source:
        print("Noise please")
        r.adjust_for_ambient_noise(source)

    while True:
        with speech_recognition.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        recognizeWithGoogle(r, audio)


if __name__ == "__main__":
    main()

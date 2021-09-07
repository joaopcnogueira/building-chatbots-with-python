import re

user_template = 'USER: {0}'
bot_template = 'BOT: {0}'

keywords = {'goodbye': ['bye', 'farewell'],
            'greet': ['hello', 'hi', 'hey'],
            'thankyou': ['thank', 'thx']}

responses = {'default': 'default message',
             'goodbye': 'goodbye for now',
             'greet': 'Hello you! :)',
             'thankyou': 'you are very welcome'}


patterns = {}
for intent, keys in keywords.items():
    patterns[intent] = re.compile('|'.join(keys))


def match_intent(message):
    matched_intent = None
    for intent, pattern in patterns.items():
        if re.search(pattern, message):
            matched_intent = intent
    return matched_intent


def respond(message):
    intent = match_intent(message)
    key = "default"
    if intent in responses:
        key = intent
    return responses[key]


def send_message(message):
    print(user_template.format(message))
    response = respond(message)
    print(bot_template.format(response))


# Send messages
send_message("hello!")
send_message("bye byeee")
send_message("thanks very much!")


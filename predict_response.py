import nltk
import json
import numpy as np
import random
import tensorflow
from data_preprocessing import get_stem_words
import pickle 

ignore = ['?', '!',',','.', "'s", "'m"]
model = tensorflow.keras.models.load_model('chatbot_model.h5')
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def preprocess_user_input(user_input):
    input_word_token_1= nltk.word_tokenize(user_input)
    input_word_token_2= get_stem_words(input_word_token_1, ignore)
    input_word_token_2= sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []

    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else: bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)
def bot_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class = np.argmax(prediction[0])
    return predicted_class
def bot_respond(user_input):
    predicted_class_label= bot_prediction(user_input)
    predicted_class = classes[predicted_class_label]
    
    for intent in intents['intents']:
        if intent['tag']==predicted_class:
            bot_respond = random.choice(intent['responses'])
            return bot_respond
print("I'm Stella, how can I help you?")

while True:
    user_input = input("Type here")
    print("Me:", user_input)

    response =  bot_respond(user_input)
    print("Stella", response)
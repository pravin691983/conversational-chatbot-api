from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse, abort, marshal, fields
import os

import nltk
from keras.models import load_model
import json
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from nltk.stem import WordNetLemmatizer
import json
import pickle
from tensorflow.keras import preprocessing


# Initialize Flask app
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# Initialise api
api = Api(app)

# Define list of books
# A List of Dicts to store all of the books
books = [{
    "id": 1,
    "title": "Zero to One",
    "author": "Peter Thiel",
    "length": 195,
    "rating": 4.17
},
    {
    "id": 2,
    "title": "Atomic Habits ",
    "author": "James Clear",
    "length": 319,
    "rating": 4.35
}
]

# load the model architecture

intents = json.loads(open('./modal_retrieval/intents.json').read())
words = pickle.load(open('./modal_retrieval/words.pkl', 'rb'))
classes = pickle.load(open('./modal_retrieval/classes.pkl', 'rb'))

# Download wordnet & punkt
nltk.download('punkt')
nltk.download('wordnet')

# Initilise Data
lemmatizer = WordNetLemmatizer()

# # Load Generative model


# def loadGenerativeModal():
#     with open('./modal_generative/ContentBase_ChatBot_model.json', 'r') as json_file:
#         json_savedModel = json_file.read()

#     # load the model architecture
#     model_content_based = tf.keras.models.model_from_json(json_savedModel)
#     model_content_based.load_weights(
#         './modal_generative/ContentBase_ChatBot_weights.hdf5')

#     return model_content_based


# model_content_based = loadGenerativeModal()
# # encoder_inputs
# # encoder_states

# # Tokenize questions & answers
# tokenizer = preprocessing.text.Tokenizer()
# maxlen_answers = 100
# maxlen_questions = 100


# def make_inference_models():

#     # Dimension for embedding layer
#     embedding_dimension = 200

#     # Dimensionality
#     dimensionality = 200  # 256

#     VOCAB_SIZE = 10000

#     encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions, ))
#     encoder_embedding = tf.keras.layers.Embedding(
#         VOCAB_SIZE, embedding_dimension, mask_zero=True)(encoder_inputs)
#     encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
#         dimensionality, return_state=True)(encoder_embedding)
#     encoder_states = [state_h, state_c]

#     # first build an encoder model with encoder inputs and encoder output states.
#     # first build an encoder model with encoder inputs and encoder output states.
#     # encoder_inputs = model_content_based.input[0]
#     # encoder_outputs, state_h_enc, state_c_enc = model_content_based.layers[2].output

#     # encoder_states = [state_h_enc, state_c_enc]
#     encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

#     # create placeholders for decoder input states
#     decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
#     decoder_state_input_c = tf.keras.layers.Input(shape=(200,))
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

#     # create new decoder states and outputs with the help of decoder LSTM and Dense layer that we trained earlier.
#     decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers,))
#     decoder_embedding = tf.keras.layers.Embedding(
#         VOCAB_SIZE, embedding_dimension, mask_zero=True)(decoder_inputs)
#     decoder_lstm = tf.keras.layers.LSTM(
#         dimensionality, return_state=True, return_sequences=True)
#     decoder_dense = tf.keras.layers.Dense(
#         VOCAB_SIZE, activation=tf.keras.activations.softmax)

#     decoder_outputs, state_h, state_c = decoder_lstm(
#         decoder_embedding, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = tf.keras.models.Model(
#         [decoder_inputs] + decoder_states_inputs,
#         [decoder_outputs] + decoder_states)

#     return encoder_model, decoder_model


# def str_to_tokens(sentence: str):
#     print("maxlen_questions", maxlen_questions)
#     print("tokenizer", tokenizer)
#     print("tokenizer.word_index", tokenizer.word_index)
#     words = sentence.lower().split()
#     # print("words", words)
#     tokens_list = list()
#     # print("Before tokens_list", tokens_list)
#     for word in words:
#         if word in tokenizer.word_index:
#             # print("tokenizer word", tokenizer.word_index[ word ])
#             tokens_list.append(tokenizer.word_index[word])
#         else:
#             tokens_list.append(tokenizer.word_index["out"])
#         # print("After tokens_list", tokens_list)
#     return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

# # predict response for generative modal


# enc_model, dec_model = make_inference_models()


# def predictResponseUsingGenerativeModal(query):
#     states_values = enc_model.predict(str_to_tokens(request))
#     empty_target_seq = np.zeros((1, 1))
#     empty_target_seq[0, 0] = tokenizer.word_index['start']
#     stop_condition = False
#     decoded_translation = ''

#     while not stop_condition:
#         dec_outputs, h, c = dec_model.predict(
#             [empty_target_seq] + states_values)
#         sampled_word_index = np.argmax(dec_outputs[0, -1, :])

#         # print("sampled_word_index", sampled_word_index)
#         if sampled_word_index == 1:
#             # print('Got end tag: Bye')
#             stop_condition = True
#             break

#         sampled_word = None
#         for word, index in tokenizer.word_index.items():
#             if sampled_word_index == index:
#                 # print("word", word)
#                 decoded_translation += ' {}'.format(word)
#                 sampled_word = word

#         if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
#             stop_condition = True

#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = sampled_word_index
#         states_values = [h, c]

#         # Print Respone
#         print(decoded_translation)

#         return decoded_translation

# # Load Retrival Modal


def loadRetrivalModal():
    with open('./modal_retrieval/RetrievalBased_ChatBot_model.json', 'r') as json_file:
        json_savedModel = json_file.read()

    # load the model architecture
    model_retrieval_based = tf.keras.models.model_from_json(json_savedModel)

    my_file = Path('./modal_retrieval/RetrievalBased_ChatBot_weights.hdf5')

    if my_file.is_file():
        # file exists
        model_retrieval_based.load_weights(
            './modal_retrieval/RetrievalBased_ChatBot_weights.hdf5')

    return model_retrieval_based


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    # print(sentence_words)
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    list_of_intents = intents_json['intents']

    if ints:
        tag = ints[0]['intent']
        # print('List of intents :', list_of_intents)
        for i in list_of_intents:
            if(i['tag'] == tag):
                result = random.choice(i['responses'])
                break
    else:
        for i in list_of_intents:
            if(i['tag'] == 'noanswer'):
                result = random.choice(i['responses'])
                break

    # print('getResponse result :', result)
    return result


def chatbot_response(msg, model_retrieval_based):
    ints = predict_class(msg, model_retrieval_based)
    print('predict_class response : ', ints)
    res = getResponse(ints, intents)
    return res


# Load model
model_retrieval_based = loadRetrivalModal()

# Schema For the Book Request JSON
chatMessageField = {
    "question": fields.String,
    "answer": fields.String
}

# Resource: Individual Book Routes


class ChatMessage(Resource):
    def __init__(self):
        # Initialize The Flsak Request Parser and add arguments as in an expected request
        self.reqparse = reqparse.RequestParser()

        self.reqparse.add_argument("question", type=str, location="json")
        self.reqparse.add_argument("answer", type=str, location="json")

    def post(self):
        args = self.reqparse.parse_args()
        print(args)
        reply = {
            "question": args["question"],
            "answer": chatbot_response(args["question"], model_retrieval_based)
        }
        print("Reply to sent .... !!!!")
        print(reply)
        # return {'Message': reply}

        return{"Predicted Response": marshal(reply, chatMessageField)}
        # return {"Reply": marshal(chatMessage, reply)}, 201


# Schema For the Book Request JSON
bookFields = {
    "id": fields.Integer,
    "title": fields.String,
    "author": fields.String,
    "length": fields.Integer,
    "rating": fields.Float
}

# Resource: Individual Book Routes


class Book(Resource):
    def __init__(self):
        # Initialize The Flsak Request Parser and add arguments as in an expected request
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument("title", type=str, location="json")
        self.reqparse.add_argument("author", type=str, location="json")
        self.reqparse.add_argument("length", type=int, location="json")
        self.reqparse.add_argument("rating", type=float, location="json")

        super(Book, self).__init__()

    # GET - Returns a single book object given a matching id
    def get(self, id):
        book = [book for book in books if book['id'] == id]

        if(len(book) == 0):
            abort(404)

        return{"book": marshal(book[0], bookFields)}

    # PUT - Given an id
    def put(self, id):
        book = [book for book in books if book['id'] == id]

        if len(book) == 0:
            abort(404)

        book = book[0]

        # Loop Through all the passed agruments
        args = self.reqparse.parse_args()
        for k, v in args.items():
            # Check if the passed value is not null
            if v is not None:
                # if not, set the element in the books dict with the 'k' object to the value provided in the request.
                book[k] = v

        return{"book": marshal(book, bookFields)}

        # Delete - Given an id
    def delete(self, id):
        book = [book for book in books if book['id'] == id]

        if(len(book) == 0):
            abort(404)

        books.remove(book[0])

        return 201


class BookList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            "title", type=str, required=True, help="The title of the book must be provided", location="json")
        self.reqparse.add_argument(
            "author", type=str, required=True, help="The author of the book must be provided", location="json")
        self.reqparse.add_argument("length", type=int, required=True,
                                   help="The length of the book (in pages)", location="json")
        self.reqparse.add_argument(
            "rating", type=float, required=True, help="The rating must be provided", location="json")

    def get(self):
        return{"books": [marshal(book, bookFields) for book in books]}

    def post(self):
        args = self.reqparse.parse_args()
        book = {
            "id": books[-1]['id'] + 1 if len(books) > 0 else 1,
            "title": args["title"],
            "author": args["author"],
            "length": args["length"],
            "rating": args["rating"]
        }

        books.append(book)
        return{"book": marshal(book, bookFields)}, 201


class HelloWorld(Resource):
    def get(self):
        # return {'Message': 'Welcome to ChatBot REST API'}
        return render_template("index.html")


# api.add_resource(HelloWorld, '/')
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/get")
# function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    predicatedResponse = chatbot_response(userText, model_retrieval_based)
    # predicatedResponse = predictResponseUsingGenerativeModal("Hello")
    # return str(englishBot.get_response(userText))
    return predicatedResponse


api.add_resource(ChatMessage, "/predict")

api.add_resource(BookList, "/books")
api.add_resource(Book, "/books/<int:id>")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)

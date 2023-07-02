from twitchBot import *
#import pickle
#from tensorflow import keras

MODEL_PATH = './model'
CHAT_PATH = 'data/chat/'
BANNED_PATH = 'data/banned/'
EXTRA_DATA = 'data/WikiQA-train.txt'
JSON_PATH = 'data/usedFiles.json'
TOKEN_PATH = 'data/tokenizer.pickle'
TFIDF_PATH = 'data/TFIDF.pickle'


twitchBot('', '', None, mode=MODE_SAVE, min=30)
async def test (tok):
  try:
      #model = keras.models.load_model(MODEL_PATH)
      print('continue')
      t = twitchBot('', '', None, min=30)
      print('continue')
      
      await (t.func(0, 0, None, 0, None, 0, 0, 0, None, None, MODE_SAVE, min))
      #asyncio.run(t.func(0, 0, newTokenizer, 0, 0, 0, 0, 0, 0, model, 'READ', min))
  except:
      print('fail')
#checkpoint = 'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'
#newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
try:
    #with open(TOKEN_PATH, 'rb') as handle:
        #tok = pickle.load(handle)
    asyncio.run(test(None))
except:
    pass

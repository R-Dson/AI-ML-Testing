from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.types import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, MessageDeletedEvent, ClearChatEvent, LeftEvent, JoinedEvent
from datetime import date
import numpy as np
#from keras.preprocessing.sequence import pad_sequences
import asyncio
import random
from transformers import AutoTokenizer
from tensorflow import keras

USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
lang = {}
cat = {}
FOLDER_PATH = 'data'

MODE_SAVE = 'Save'
MODE_READ = 'Read'

ALLOWED = 0
BANNED = 1

class twitchBot:

    def __init__(self, id, secret, model, mode=MODE_READ, min=30):
        self.id = id
        self.secret = secret
        if mode == MODE_SAVE:
            print('test')
            #asyncio.run(self.func(0, 0, 0, 0, 0, 0, 0, 0,0, model, mode, min))
        #await (self.func(MAX_LEN, maxLenPost, tokenizer, tokenizerLang, sequencesChar, tokenizerCharLongest, maxLenChar, numChar, cv, model, mode, min))
        """
        if mode == MODE_READ:
            
        elif mode == MODE_SAVE:
            asyncio.run(self.func(MAX_LEN, mode, min))"""

    async def func(self, MAX_LEN, maxLenPost, tokenizer, tokenizerLang, sequencesChar, tokenizerCharLongest, maxLenChar, numChar, cv, model, mode, min):
        tc = _TwitchChat(self.id, self.secret, mode, MAX_LEN, maxLenPost, tokenizer, tokenizerLang, sequencesChar, tokenizerCharLongest, maxLenChar, numChar, cv, model)

        while True:
            if mode == MODE_READ:
                await tc.start()
            elif mode == MODE_SAVE:
                await tc.start(saveMessages=True, saveDelete=True)
            await asyncio.sleep(60*min)

class _TwitchChat:
    def __init__(self, id, secret, mode, MAX_LEN, maxLenPost, tokenizer, tokenizerLang, sequencesChar, tokenizerCharLongest, maxLenChar, numChar, cv, model):
        self.id = id
        self.secret = secret
        self.mode = mode
        self.i = 0
        self.j = 0

        self.correct = 0
        self.wrong = 0

        if mode == MODE_SAVE:
            pass
        
        elif mode == MODE_READ:
            self.MAX_LEN = MAX_LEN
            self.tokenizer = newTokenizer
            self.tokenizerLang = tokenizerLang
            self.model = model
            self.sequencesChar = sequencesChar
            self.maxLenPost = maxLenPost
            self.tokenizerChar = tokenizerCharLongest
            self.maxLenChar = maxLenChar
            self.numChar = numChar
            self.cv = cv

    def writeToFile(self, fileName, msgTxt, streamerName, prefix = ''):        
        wText = '{msgS}\t{catS}\t{langS}\t{sn}\n'.format(msgS = msgTxt, catS = cat[streamerName], langS = lang[streamerName], sn=streamerName)
        filePath = FOLDER_PATH + '/'+ prefix + str(date.today()) + '-' + fileName
        try:
            file = open(filePath, mode="a")
        except:
            file = open(filePath, mode="x")
        print(wText)
        file.write(wText)
        file.close()
        
    async def clearStreamers(self):
        for i in range(0, len(self.TARGET_CHANNELS), 19):
            await self.chat.leave_room(self.TARGET_CHANNELS[i:i+19])
        self.TARGET_CHANNELS = []
    
    async def searchStreams(self, numStreams):
        async for x in self.twitch.get_streams(first=numStreams):
            #print(x.user_name)
            x.user_name.replace(' ', '')
            if ' ' not in x.user_name:
                x.user_name = x.user_name.lower()
                cat[x.user_name] = x.game_id
                lang[x.user_name] = x.language
                self.TARGET_CHANNELS.append(x.user_name)
            else:
                print(f'skipping: {x.user_name}')
            if len(self.TARGET_CHANNELS) > numStreams:
                self.TARGET_CHANNELS = list(set(self.TARGET_CHANNELS))
                break
        self.TARGET_CHANNELS = list(set(self.TARGET_CHANNELS))
    
    async def updater(self, min):
        print('updater...')
        while True: # some drift is fine.. might fix later
            await asyncio.sleep(60*min)
            if self.running:
                print('updating streams')
                await self.clearStreamers()
                print('searching streams')
                await self.searchStreams()
                print('joining streams')
                await self.joinRooms()
    
    async def joinRooms(self):
        for i in range(0, len(self.TARGET_CHANNELS), 19):
            try:
                await self.chat.join_room(self.TARGET_CHANNELS[i:i+19])
            except:
                print('Failed to join rooms.')
            
    async def on_ready(self, ready_event: EventData):
        await self.joinRooms()
        print('connected...\n')
    
    async def on_deleted(self, msg: MessageDeletedEvent):
        if lang[msg.room.name] != 'en':
                return
        
        if self.mode == MODE_SAVE:
            self.writeToFile("banned.txt", msg.message, msg.room.name, prefix='banned/')

        elif self.mode == MODE_READ:

            X = [msg.message]
            sequences = newTokenizer(X, padding = 'max_length')
            sequences = np.array(sequences.data['input_ids'])
            #sequences = self.tokenizer.texts_to_sequences(X)
            #sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=self.MAX_LEN)
            #sequencesChar = self.tokenizerChar.texts_to_sequences(X)
            #sequencesChar = pad_sequences(sequencesChar, padding='post', maxlen=self.maxLenChar)
            #sequencesFreq = self.cv.transform(X).toarray()

            inputs = sequences#sequencesFreq#[sequences, sequencesFreq]

            """
        
            Xt = self.tokenizer.texts_to_sequences([msg.message])
            sequencesChar = self.tokenizerChar.texts_to_sequences([msg.message])
            t = self.cv.transform([msg.message]).toarray()

            Xp = pad_sequences(Xt, padding='post', maxlen=self.MAX_LEN)
            Xp = Xp[0]
            c = np.array(list(Xp) + sequencesChar[0], dtype=np.int64)

            padLength = self.maxLenChar + self.MAX_LEN
            Xp = pad_sequences([c], padding='post', maxlen=padLength)

            a = np.array(Xp[0], dtype=np.float64)
            t = self.cv.transform([msg.message]).toarray()
            b = np.array(t, dtype=np.float64)[0]
            c = np.array([np.concatenate((a, b))], dtype=np.float64)"""
            
            
            try:
                s = self.model.predict(inputs, verbose=False)
                values = s[0]
                if np.argmax(values) == 1:
                    self.wrong = self.wrong + 1
                    print(f'{self.wrong} Not deleted: {msg.message}')
                    return
                else:
                    self.correct = self.correct + 1
                    print(f'{self.correct} Correct deleted: {msg.message}')
                    #print(f'{msg.message} : {s}')
            except:
                print(('Error predicting: {msg.message}'))

            
            
    async def on_cleared(self, msg: ClearChatEvent):
        pass
    
    async def on_joined(self, ready_event: JoinedEvent):
        print(f'joined: {ready_event.room_name}')

    async def on_message(self, msg: ChatMessage):
        #print(f'in {msg.room.name}, {msg.user.name} said: {msg.text}')
        #self.writeToFile(msg.text)
        if lang[msg.room.name] != 'en':
                return

        if self.mode == MODE_SAVE:
            if random.random() > 0.99:
                self.writeToFile("chat.txt", msg.text, msg.room.name, prefix='chat/')

        elif self.mode == MODE_READ:
            X = [msg.text]
            #sequences = self.tokenizer.texts_to_sequences(X)
            #sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=self.MAX_LEN)
            #sequencesChar = self.tokenizerChar.texts_to_sequences(X)
            #sequencesChar = pad_sequences(sequencesChar, padding='post', maxlen=self.maxLenChar)
            #sequencesFreq = self.cv.transform(X).toarray()
            #sequences = newTokenizer(X, padding = 'max_length')
            #sequences = np.array(sequences.data['input_ids'])
            sequences = newTokenizer(X, padding='max_length', max_length=512, truncation=True)
            sequences = np.array(sequences['input_ids'])
            inputs = sequences#[sequences, sequencesFreq]#sequencesChar, 
            #print(inputs)
            try:
                s = self.model.predict(inputs, verbose=False)
                #print(s)
                values = s[0]
                #ind = np.array(values).argmax()
                self.i += 1
                #if max(values) > 0.11:
                #print(max(values))

                #if max(values) < 0.5:
                    #if max(values) > 0.1:
                        #print(max(values))
                #    return
                #print(values[np.argmax(values)])
                #print(values)
                
                if max(values) > .5:#np.argmax(values) == 0:
                    self.j += 1
                    f = self.j/self.i
                    print(f'ratio: {self.j} / {self.i} = {f}. s: {s}. {msg.text}')
            except:
                print('Error predicting: ' + msg.text)
                
            return
            #Xt = self.tokenizer.texts_to_sequences([msg.text])
            #sequencesChar = self.tokenizerChar.texts_to_sequences([msg.text])
            #t = self.cv.transform([msg.text]).toarray()
            #language = lang[msg.room.name]
            #l = self.tokenizerLang.texts_to_sequences([language])
            #Xp = pad_sequences(Xt, padding='post', maxlen=self.MAX_LEN)
            #Xp = Xp[0]
            """               
                w = X[j]
                for s in w.split():
                    indx = tokenizer.word_index[s]
                    wc = tokenizer.word_counts[s]
                    wclist.append(wc)
                mergedseq.append(c)
                #seqmerged.append([sequences[j]] + list(sequencesChar[j]))
                pass"""

            #c = np.array(list(Xp) + sequencesChar[0], dtype=np.int64)
            
            #padLength = self.maxLenChar + self.MAX_LEN
            #Xp = pad_sequences([c], padding='post', maxlen=padLength)

            mergedseq = []
            #a = np.array(Xp[0], dtype=np.float64)
            #t = self.cv.transform([msg.text]).toarray()
            #b = np.array(t, dtype=np.float64)[0]
            #c = np.array([np.concatenate((a, b))], dtype=np.float64)
            c = newTokenizer([msg.text], padding = 'max_length')
            #s = self.model.predict(c, verbose=False)
            #try:
                
            #except:
            #    print(f('Error predicting: {msg.text}'))
            values = s[0]
            #ind = np.array(values).argmax()
            self.i += 1
            #print(max(values))
            if max(values) < 0.5:
                #if max(values) > 0.1:
                    #print(max(values))
                return

            if BANNED == BANNED:
                self.j += 1
                f = self.j/self.i
                print(f'ratio: {self.j} / {self.i} = {f}. s: {s}. {msg.text}')
    
    async def on_leaving(self, ready_event: LeftEvent):
        print(f'left: {ready_event.room_name}')
    
    async def start(self, saveDelete=False, saveMessages=False):
        self.TARGET_CHANNELS = []
        self.twitch = await Twitch(self.id, self.secret)
        
        await self.auth()
        
        await self.searchStreams(100)
        
        chat = await Chat(self.twitch)
        self.chat = chat
        
        chat.register_event(ChatEvent.READY, self.on_ready)

        if saveDelete or self.mode == MODE_READ:
            chat.register_event(ChatEvent.MESSAGE_DELETE, self.on_deleted)
        if saveMessages or self.mode == MODE_READ:
            chat.register_event(ChatEvent.MESSAGE, self.on_message)
        
        chat.register_event(ChatEvent.JOINED, self.on_joined)
        chat.register_event(ChatEvent.LEFT, self.on_leaving)
        #chat.register_event(ChatEvent.CHAT_CLEARED, self.on_cleared)
    
        chat.start()
        self.running = True
        
    async def stop(self):
        self.clearStreamers()
        self.running = False
        
    async def auth(self):
        print('auth')
        auth = UserAuthenticator(self.twitch, [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT])
        print('auth2')
        token, refresh_token = await auth.authenticate()
        await self.twitch.set_user_authentication(token, USER_SCOPE, refresh_token)

async def test ():
  try:
      model = keras.models.load_model('./model')
      print('continue')
      t = twitchBot('', '', model, min=30)
      print('continue')
      
      await (t.func(0, 0, newTokenizer, 0, None, 0, 0, 0, None, model, MODE_READ, min))
      #asyncio.run(t.func(0, 0, newTokenizer, 0, 0, 0, 0, 0, 0, model, 'READ', min))
  except:
      print('fail')
#await test()
checkpoint = 'Epidot/TwitchLeagueBert-1000k'#'bert-base-uncased'
newTokenizer = AutoTokenizer.from_pretrained(checkpoint)
asyncio.run(test())
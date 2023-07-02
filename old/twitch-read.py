from twitchAPI.twitch import Twitch
#from twitchAPI.helper import first
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.types import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, MessageDeletedEvent, ClearChatEvent, LeftEvent, JoinedEvent
from datetime import date
import asyncio
import random

USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
lang = {}
cat = {}
FOLDER_PATH = 'data'

class TwitchChat:
    
    def __init__(self, id, secret):
        self.id = id
        self.secret = secret
        
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
            await self.chat.join_room(self.TARGET_CHANNELS[i:i+19])
            
    async def on_ready(self, ready_event: EventData):
        await self.joinRooms()
        print('connected...\n')
    
    async def on_deleted(self, msg: MessageDeletedEvent):
        self.writeToFile("banned.txt", msg.message, msg.room.name, prefix='banned/')
        
    async def on_cleared(self, msg: ClearChatEvent):
        pass
    
    async def on_joined(self, ready_event: JoinedEvent):
        print(f'joined: {ready_event.room_name}')

    async def on_message(self, msg: ChatMessage):
        #print(f'in {msg.room.name}, {msg.user.name} said: {msg.text}')
        #self.writeToFile(msg.text)
        if lang[msg.room.name] != 'en':
            return
        if random.random() > 0.999:
            self.writeToFile("chat.txt", msg.text, msg.room.name, prefix='chat/')
    
    async def on_leaving(self, ready_event: LeftEvent):
        print(f'left: {ready_event.room_name}')
    
    async def start(self, saveDelete=True, saveMessages=False):
        self.TARGET_CHANNELS = []
        self.twitch = await Twitch(self.id, self.secret)
        
        await self.auth()
        
        await self.searchStreams(100)
        
        chat = await Chat(self.twitch)
        self.chat = chat
        
        chat.register_event(ChatEvent.READY, self.on_ready)
        if saveDelete:
            chat.register_event(ChatEvent.MESSAGE_DELETE, self.on_deleted)
        if saveMessages:
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
        auth = UserAuthenticator(self.twitch, [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT])
        token, refresh_token = await auth.authenticate()
        await self.twitch.set_user_authentication(token, USER_SCOPE, refresh_token)


async def func(min=1):
    while True:
        tc = TwitchChat('qjrrpzrombru7ghf5nf4pjmlgr6uka', '1x8fy8co8dbhdjk3m8iey4zof57pbt')
        #await asyncio.run(tc.start())
        await tc.start(saveMessages=True)
        await asyncio.sleep(60*min)
        
asyncio.run(func(30))

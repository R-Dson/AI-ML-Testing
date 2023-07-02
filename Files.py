#import Model
from Run import TEXT, CATEGORY, LANG, NAME, VALUE, ALLOWED

class File:
    def __init__(self, fileName):
        b = open(fileName, "r")
        self.bl = b.readlines()
        self.text =  []
        self.cat = []
        self.lang = []
        self.name = []
        self.value = []
        self.question = []
        self.ans = []
    
    def split(self, filterList=True):
        for line in self.bl:
            lSplit = line.split('\t')
            try:
                t = lSplit[TEXT]
                if True:#(t not in self.text) or filterList == False:
                    c = int(lSplit[CATEGORY])
                    l = lSplit[LANG]
                    n = lSplit[NAME]
                    v = int(lSplit[VALUE])
                    
                    self.text.append(t)
                    self.cat.append(c)
                    self.lang.append(l)
                    self.name.append(n)
                    self.value.append(v)
            except:
                pass
                #print('failed')

    def splitExtra(self):
        for line in self.bl:
            lSplit = line.split('\t')
            try:
                q = lSplit[0]
                a = lSplit[1]
                
                if True:# q not in self.text:
                    self.text.append(q)
                    self.value.append(ALLOWED)
                if True:# a not in self.text:
                    self.text.append(a)
                    self.value.append(ALLOWED)
            except:
                print('Error loading text.')

    def splitExtraChat(self):
        return [ x.split(': ')[1][:-1] for x in self.bl]
        for line in self.bl:
            lSplit = line.split(': ')
            try:
                text = lSplit[1]
            except:
                pass
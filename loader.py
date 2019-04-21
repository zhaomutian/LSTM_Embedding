import re
def base_read_file(file_name='1_三俭草.txt'):#return list of list of tuple [   [(word,tag),(word,tag)...] ...]
    words_tags=[]
    with open(file_name,'r') as fd:
        str_all=fd.readlines()[0]
        word_tokens=str_all.split()
        for token in word_tokens:
            #token= token.decode('utf8')
            char_and_tag=re.findall(u"([\u4e00-\u9fa5]|，|。|[0-9]*)/([A-Z]-[a-z]+|O)",token)
            #print(char_and_tag)
            words_tags.append(char_and_tag)
    return words_tags

class Loader:

    def __init__(self,file_name='1_三俭草.txt'):  # return list of list of tuple [   [(word,tag),(word,tag)...] ...]
        words_tags = []
        with open(file_name, 'r') as fd:
            str_all = fd.readlines()
            word_tokens=[]
            for line in str_all:
                word_tokens.extend(line.split())
            #word_tokens = str_all.split()
            for token in word_tokens:
                # token= token.decode('utf8')
                char_and_tag = re.findall(u"([\u4e00-\u9fa5]|，|。|[0-9]*)/([A-Z]-[a-z]+|O)", token)
                # print(char_and_tag)
                words_tags.append(char_and_tag)
        self.file=file_name
        self.raw_words_tags=words_tags


    def get_chars_list(self):
        l=[]
        for i in self.raw_words_tags:
            for (w,t) in i:
                l.append(w)
        return l

    def get_dis_chars_list(self):
        return list(set(self.get_chars_list(self)))






if __name__=='__main__':
    pass

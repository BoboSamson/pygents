import abc
from pygents.util import count_subelements, dictcount, calc_f1, counters_init
from pygents.text import preprocess_text, grams_count_with_char_freedoms

# Basic Tokenizer
class Tokenizer(abc.ABC):

    def __init__(self, debug=True):
        self.debug = debug

    def tokenize(self,text):
        return text.split()

assert str(Tokenizer().tokenize("ab c")) == "['ab', 'c']"

def tokenize_detaching_head(text,chars="'\"{[("):
    tokens = []
    for head in range(len(text)):
        found = chars.find(text[head])
        if found >= 0:
            tokens.append(chars[found])
        else:
            return tokens, text[head:]
    return tokens, None
assert str(tokenize_detaching_head("test")) == "([], 'test')"
assert str(tokenize_detaching_head("'\"")) == '(["\'", \'"\'], None)'
assert str(tokenize_detaching_head("\"'test")) == "(['\"', \"'\"], 'test')"


def tokenize_detaching_tail(text,chars="'\":,;.!?}])"):
    tokens = []
    length = len(text)
    for i in range(length):
        tail = length - i - 1
        found = chars.find(text[tail])
        if found >= 0:
            tokens.append(chars[found])
        else:
            tokens.reverse()
            return tokens, text[:tail + 1]
    tokens.reverse()
    return tokens, None
assert str(tokenize_detaching_tail("test")) == "([], 'test')"
assert str(tokenize_detaching_tail("test'")) == "([\"'\"], 'test')"
assert str(tokenize_detaching_tail("test.\"")) == "(['.', '\"'], 'test')"
assert str(tokenize_detaching_tail("test').\"")) == "([\"'\", ')', '.', '\"'], 'test')"

    
def tokenize_split_with_delimiters_and_quotes(text):
    tokens = []
    splits = text.split(' ')
    for split in splits:
        if len(tokens) > 0:
            tokens.append(' ')
        head, token = tokenize_detaching_head(split)
        tokens.extend(head)
        if token is not None and len(token) > 0: 
            tail, token = tokenize_detaching_tail(token)
            if token is not None and len(token) > 0:
                tokens.append(token)
            tokens.extend(tail)
    return tokens
assert str(tokenize_split_with_delimiters_and_quotes("man says hi")) == "['man', ' ', 'says', ' ', 'hi']"
assert str(tokenize_split_with_delimiters_and_quotes("man (tom) says 'hi there!' to me.")) == "['man', ' ', '(', 'tom', ')', ' ', 'says', ' ', \"'\", 'hi', ' ', 'there', '!', \"'\", ' ', 'to', ' ', 'me', '.']"



# Exttended Tokenizer based on "freedoms"
class FreedomTokenizer(Tokenizer):

    def __init__(self, name=None, max_n=7, mode='grams', debug=False):
        Tokenizer.__init__(self,debug=debug)
        self.max_n = max_n
        self.model = pickle.load(open(name, 'rb')) if name is not None else [{},{},{}]
        self.mode = mode

    def train(self,texts,max_n=None):
        if max_n is None:
            max_n = self.max_n
        model = counters_init(max_n) 
        for text in texts:
            text = preprocess_text(text)
            if self.mode == 'grams':
                for n in range(max_n):
                    grams_count_with_gram_freedoms(model,text,n+1,debug=self.debug)
            else: # 'chars' - legacy
                chars = list(text)
                for n in range(max_n):
                    grams_count_with_char_freedoms(model[0],model[1],model[2],chars,n+1,debug=self.debug)
        #merge n-specific models into joint ones
        for i in range(3):
            for d in model[i]:
                self.model[i].update(d)
        return self
        
    def tokenize(self,text):
        #TODO
        return text.split()

    def count_params(self):
        return count_subelements(self.model)
    
    def store(self,path):
        pickle.dump(self.model, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

    
_test_tokenizer = FreedomTokenizer(max_n=2,mode='chars',debug=False).train(["pig"])
assert _test_tokenizer.count_params() == 11
assert str(_test_tokenizer.model) == "[{'p': 1, 'i': 1, 'g': 1, 'pi': 1, 'ig': 1}, {'p': {'i': 1}, 'i': {'g': 1}, 'pi': {'g': 1}}, {'i': {'p': 1}, 'g': {'i': 1}, 'ig': {'p': 1}}]"
_test_tokenizer = FreedomTokenizer(max_n=2,mode='chars').train(["ding","dong"])
#print(_test_tokenizer.count_params())
assert _test_tokenizer.count_params() == 28
#print(str(_test_tokenizer.model[0]))
#print(str(_test_tokenizer.model[1]))
#print(str(_test_tokenizer.model[2]))
#print(str(_test_tokenizer.model))
assert str(_test_tokenizer.model) == "[{'d': 2, 'i': 1, 'n': 2, 'g': 2, 'o': 1, 'di': 1, 'in': 1, 'ng': 2, 'do': 1, 'on': 1}, {'d': {'i': 1, 'o': 1}, 'i': {'n': 1}, 'n': {'g': 2}, 'o': {'n': 1}, 'di': {'n': 1}, 'in': {'g': 1}, 'do': {'n': 1}, 'on': {'g': 1}}, {'i': {'d': 1}, 'n': {'i': 1, 'o': 1}, 'g': {'n': 2}, 'o': {'d': 1}, 'in': {'d': 1}, 'ng': {'i': 1, 'o': 1}, 'on': {'d': 1}}]"


def tokenize_with_opposite_metrics(model,text,back,forw,nlist,threshold=0.5,debug=False):
    tokens = []
    token = ''
    df = profile_freedoms_avg_df(model,text,[forw,back],nlist)
    length = len(df)
    for i in range(length):
        iplus1 = i+1
        brk_back = True if iplus1 < length and df.loc[iplus1][back] >= threshold else False
        brk_forw = True if df.loc[i][forw] >= threshold else False
        token += df.loc[i]['gram']
        if debug:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(df.loc[i]['gram'],'-' if brk_back else '', '+' if brk_forw else '',round(df.loc[i][back],2),round(df.loc[i][forw],2),token))
        if len(token) > 0 and (brk_back or brk_forw):
            tokens.append(token)
            token = ''
    if len(token) > 0:
            tokens.append(token)
    return tokens

 
def profile_freedoms_ex_df(model,text,n,debug=False):
    df = pd.DataFrame(profile_freedoms(model,text,n,debug=debug),columns=['pos','gram','f+','f-'])
    df['ddf+'] = (df['f+'] - df['f+'].mean()).clip(lower=0)
    df['ddf-'] = (df['f-'] - df['f-'].mean()).clip(lower=0)
    df['ddf+|ddf-'] = df['ddf+'] + df['ddf-'].shift(-1)
    df['ddf+&ddf-'] = df['ddf+'] * df['ddf-'].shift(-1)
    df['df+'] = df['f+'].diff() 
    df['df-'] = -df['f-'].diff().shift(-1)
    df['df+|df-'] = df['df+'] + df['df-']
    df['df+&df-'] = df['df+'] * df['df-']
    # We assigned a “peak” value to each character transition, 
    # computed by adding the value of the preceding increase in freedom to the following decrease in freedom. 
    # We characterized token boundaries based on the sum of their forward- and backward-reading peak values.
    df['peak+'] = df['df+'] - df['df+'].shift(-1)
    df['peak-'] = df['df-'] - df['df-'].shift(1)
    df['f+|f-'] = df['f+'] + df['f-'].shift(-1)
    df['f+&f-'] = df['f+'] * df['f-'].shift(-1)
    return df


def profile_freedoms_avg_df(model,text,metrics,nlist,debug=False):
    res_df = None
    for n in nlist:
        df = profile_freedoms_ex_df(model,text,n)
        if res_df is None:
            res_df = df[['pos','gram']+metrics].copy()
        else:
            for m in metrics:
                res_df[m] = res_df[m] + df[m]
    for m in metrics:
        res_df[m] = res_df[m]/res_df[m].max()
    return res_df




import json
from enum import Enum, auto


class TokenType(Enum):
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COLON = auto()
    COMMA = auto()
    STRING = auto()
    NUMBER = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    EOF = auto()
    IDENTIFIER = auto()


class Token:
    def __init__(self, type, value, pos):
        self.type = type
        self.value = value
        self.pos = pos


class LexerState(Enum):
    START = auto()
    IN_STRING = auto()
    IN_ESCAPE = auto()
    IN_NUMBER = auto()
    IN_IDENTIFIER = auto()
    IN_LINE_COMMENT = auto()
    IN_BLOCK_COMMENT = auto()
    BLOCK_COMMENT_STAR = auto()


class JSONLexer:
    def __init__(self):
        self.text = ""
        self.pos = 0
        self.state = LexerState.START
        self.current_token = []
        self.quote_char = None
        
    def feed(self, text):
        self.text = text
        self.pos = 0
        self.state = LexerState.START
        self.current_token = []
        
    def peek(self, offset=0):
        idx = self.pos + offset
        if idx < len(self.text):
            return self.text[idx]
        return None
        
    def advance(self):
        if self.pos < len(self.text):
            c = self.text[self.pos]
            self.pos += 1
            return c
        return None
        
    def tokenize(self):
        tokens = []
        
        while self.pos < len(self.text):
            if self.state == LexerState.START:
                c = self.peek()
                
                if c in ' \t\n\r':
                    self.advance()
                    continue
                    
                if c == '/':
                    next_c = self.peek(1)
                    if next_c == '/':
                        self.advance()
                        self.advance()
                        self.state = LexerState.IN_LINE_COMMENT
                        continue
                    elif next_c == '*':
                        self.advance()
                        self.advance()
                        self.state = LexerState.IN_BLOCK_COMMENT
                        continue
                        
                if c == '{':
                    tokens.append(Token(TokenType.LBRACE, c, self.pos))
                    self.advance()
                elif c == '}':
                    tokens.append(Token(TokenType.RBRACE, c, self.pos))
                    self.advance()
                elif c == '[':
                    tokens.append(Token(TokenType.LBRACKET, c, self.pos))
                    self.advance()
                elif c == ']':
                    tokens.append(Token(TokenType.RBRACKET, c, self.pos))
                    self.advance()
                elif c == ':':
                    tokens.append(Token(TokenType.COLON, c, self.pos))
                    self.advance()
                elif c == ',':
                    tokens.append(Token(TokenType.COMMA, c, self.pos))
                    self.advance()
                elif c in '"\'':
                    self.quote_char = c
                    self.current_token = []
                    self.state = LexerState.IN_STRING
                    self.advance()
                elif c == '-' or c.isdigit():
                    self.current_token = [c]
                    self.state = LexerState.IN_NUMBER
                    self.advance()
                elif c.isalpha() or c == '_':
                    self.current_token = [c]
                    self.state = LexerState.IN_IDENTIFIER
                    self.advance()
                else:
                    self.advance()
                    
            elif self.state == LexerState.IN_STRING:
                c = self.peek()
                if c is None:
                    tokens.append(Token(TokenType.STRING, ''.join(self.current_token), self.pos))
                    self.state = LexerState.START
                    break
                elif c == '\\':
                    self.state = LexerState.IN_ESCAPE
                    self.advance()
                elif c == self.quote_char:
                    tokens.append(Token(TokenType.STRING, ''.join(self.current_token), self.pos))
                    self.advance()
                    self.state = LexerState.START
                    self.current_token = []
                else:
                    self.current_token.append(c)
                    self.advance()
                    
            elif self.state == LexerState.IN_ESCAPE:
                c = self.peek()
                if c is None:
                    self.state = LexerState.IN_STRING
                    break
                escape_map = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', 'f': '\f'}
                if c in escape_map:
                    self.current_token.append(escape_map[c])
                else:
                    self.current_token.append(c)
                self.advance()
                self.state = LexerState.IN_STRING
                
            elif self.state == LexerState.IN_NUMBER:
                c = self.peek()
                if c and (c.isdigit() or c in '.eE+-'):
                    self.current_token.append(c)
                    self.advance()
                else:
                    num_str = ''.join(self.current_token)
                    tokens.append(Token(TokenType.NUMBER, num_str, self.pos))
                    self.current_token = []
                    self.state = LexerState.START
                    
            elif self.state == LexerState.IN_IDENTIFIER:
                c = self.peek()
                if c and (c.isalnum() or c == '_'):
                    self.current_token.append(c)
                    self.advance()
                else:
                    ident = ''.join(self.current_token)
                    if ident == 'true':
                        tokens.append(Token(TokenType.TRUE, True, self.pos))
                    elif ident == 'false':
                        tokens.append(Token(TokenType.FALSE, False, self.pos))
                    elif ident == 'null':
                        tokens.append(Token(TokenType.NULL, None, self.pos))
                    else:
                        tokens.append(Token(TokenType.IDENTIFIER, ident, self.pos))
                    self.current_token = []
                    self.state = LexerState.START
                    
            elif self.state == LexerState.IN_LINE_COMMENT:
                c = self.peek()
                if c is None or c == '\n':
                    self.state = LexerState.START
                    if c:
                        self.advance()
                else:
                    self.advance()
                    
            elif self.state == LexerState.IN_BLOCK_COMMENT:
                c = self.peek()
                if c is None:
                    self.state = LexerState.START
                    break
                elif c == '*':
                    self.state = LexerState.BLOCK_COMMENT_STAR
                    self.advance()
                else:
                    self.advance()
                    
            elif self.state == LexerState.BLOCK_COMMENT_STAR:
                c = self.peek()
                if c is None:
                    self.state = LexerState.START
                    break
                elif c == '/':
                    self.state = LexerState.START
                    self.advance()
                elif c == '*':
                    self.advance()
                else:
                    self.state = LexerState.IN_BLOCK_COMMENT
                    self.advance()
        
        if self.state == LexerState.IN_STRING:
            tokens.append(Token(TokenType.STRING, ''.join(self.current_token), self.pos))
        elif self.state == LexerState.IN_NUMBER:
            tokens.append(Token(TokenType.NUMBER, ''.join(self.current_token), self.pos))
        elif self.state == LexerState.IN_IDENTIFIER:
            ident = ''.join(self.current_token)
            if ident == 'true':
                tokens.append(Token(TokenType.TRUE, True, self.pos))
            elif ident == 'false':
                tokens.append(Token(TokenType.FALSE, False, self.pos))
            elif ident == 'null':
                tokens.append(Token(TokenType.NULL, None, self.pos))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, ident, self.pos))
                
        tokens.append(Token(TokenType.EOF, None, self.pos))
        return tokens


class JSONParser:
    def __init__(self):
        self.tokens = []
        self.pos = 0
        
    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, None, -1)
        
    def peek(self, offset=1):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token(TokenType.EOF, None, -1)
        
    def advance(self):
        if self.pos < len(self.tokens):
            self.pos += 1
            
    def parse(self, tokens):
        self.tokens = tokens
        self.pos = 0
        return self.parse_value()
        
    def parse_value(self):
        token = self.current()
        
        if token.type == TokenType.LBRACE:
            return self.parse_object()
        elif token.type == TokenType.LBRACKET:
            return self.parse_array()
        elif token.type == TokenType.STRING:
            self.advance()
            return token.value
        elif token.type == TokenType.NUMBER:
            self.advance()
            val = token.value
            if '.' in val or 'e' in val or 'E' in val:
                return float(val)
            return int(val)
        elif token.type == TokenType.TRUE:
            self.advance()
            return True
        elif token.type == TokenType.FALSE:
            self.advance()
            return False
        elif token.type == TokenType.NULL:
            self.advance()
            return None
        elif token.type == TokenType.IDENTIFIER:
            self.advance()
            return token.value
        else:
            return None
            
    def parse_object(self):
        obj = {}
        self.advance()
        
        while self.current().type != TokenType.RBRACE and self.current().type != TokenType.EOF:
            key_token = self.current()
            
            if key_token.type == TokenType.STRING:
                key = key_token.value
                self.advance()
            elif key_token.type == TokenType.IDENTIFIER:
                key = key_token.value
                self.advance()
            elif key_token.type == TokenType.COMMA:
                self.advance()
                continue
            else:
                break
                
            if self.current().type == TokenType.COLON:
                self.advance()
            elif self.current().type in (TokenType.STRING, TokenType.NUMBER, TokenType.LBRACE, 
                                         TokenType.LBRACKET, TokenType.TRUE, TokenType.FALSE, 
                                         TokenType.NULL, TokenType.IDENTIFIER):
                pass
            else:
                break
                
            value = self.parse_value()
            obj[key] = value
            
            if self.current().type == TokenType.COMMA:
                self.advance()
                if self.current().type == TokenType.RBRACE:
                    break
            elif self.current().type == TokenType.RBRACE:
                break
            elif self.current().type in (TokenType.STRING, TokenType.IDENTIFIER):
                continue
                
        if self.current().type == TokenType.RBRACE:
            self.advance()
            
        return obj
        
    def parse_array(self):
        arr = []
        self.advance()
        
        while self.current().type != TokenType.RBRACKET and self.current().type != TokenType.EOF:
            if self.current().type == TokenType.COMMA:
                self.advance()
                if self.current().type == TokenType.RBRACKET:
                    break
                continue
                
            value = self.parse_value()
            arr.append(value)
            
            if self.current().type == TokenType.COMMA:
                self.advance()
                if self.current().type == TokenType.RBRACKET:
                    break
            elif self.current().type == TokenType.RBRACKET:
                break
                
        if self.current().type == TokenType.RBRACKET:
            self.advance()
            
        return arr


class JSONRepair:
    def __init__(self):
        self.lexer = JSONLexer()
        self.parser = JSONParser()
        
    def parse(self, text):
        self.lexer.feed(text)
        tokens = self.lexer.tokenize()
        result = self.parser.parse(tokens)
        return result
        
    def parse_stream(self, stream):
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        return self.parse(''.join(chunks))


if __name__ == "__main__":
    repair = JSONRepair()
    
    tests = [
        '{"name": "John", "age": 30}',
        '{name: "John", age: 30}',
        '{"name": "John" "age": 30}',
        "{'name': 'John', 'age': 30}",
        '{"name": "John", "age": 30,}',
        '[1, 2, 3,]',
        '[1 2 3]',
        '{"user": {name: "John", age: 30}, "active": true}',
        '{"name": "John" /* comment */ "age": 30}',
        '{"name": "John", // comment\n"age": 30}',
        '{name: "John" age: 30 city: "NYC"}',
        '{"incomplete": "string}',
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test}")
        try:
            result = repair.parse(test)
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
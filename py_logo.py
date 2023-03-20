"""
main module to interpret a language

this language will be an interpreted language

features that the language will support:
- custom error messages
- bindings (we will have to implement a bindings dicctionary)
- typed bindings
- functions
- for loops
- multi-line commands

NOTE: log custom error with most recursion profundity since it is the one
      to most likely help the user

TODO: custom error messages, find a way to deal with exceptions

"""

from typing import List, Dict, Optional, Union, Callable
import turtle
import sys
from pathlib import Path

from typing import List
from enum import Enum
import re

# reg exes representing valid instances of tokens

class TokenType(Enum):
    # not asociated to a value 
    LET_KEYWORD = 'let'
    EQUAL_OPERATOR = '='
    COLON = '\:'
    OPEN_COMMENT = '/\*'
    CLOSE_COMMENT = '\*/'
    FORWARD_MOVE = 'dela'
    BACKWARDS_MOVE = 'atra'
    LEFT_TURN = 'izqd'
    RIGHT_TURN = 'dere'
    PEN_UP = 'lplum'
    PEN_DOWN = 'bplum'
    CHANGE_COLOUR = 'colp'
    CENTER = 'cntr'
    CLEAR = 'limp'
    FOR_LOOP = 'rept'
    DELIMITATOR = ';'
    OPEN_SCOPE = '{'
    CLOSE_SCOPE = '}'
    COLOUR_DATA_TYPE = 'colour'
    NUMBER_DATA_TYPE = 'number'
    
    # asociated to a value
    
    # first to be returned before a binding name
    COLOUR_LITERAL = 'black|red|yellow|green|orange|blue'
    NUMBER_LITERAL = '([1-9][0-9]*| 0)(.)?([0-9])*'
    
    BINDING_NAME = '[A-Za-z]([A-Za-z0-9_])*'
    

    # not matched any of the above
    UNKNOWN = 'unknown'
    EOF = '@@EOF@@' # special token to indicate end of file

value_associated_tokens = [
    TokenType.UNKNOWN, 
    TokenType.BINDING_NAME,
    TokenType.NUMBER_LITERAL,
    TokenType.COLOUR_LITERAL
]

class Token:
    def __init__(self, token_type: TokenType, value):
        self.token_type = token_type 
        
        if token_type == TokenType.NUMBER_LITERAL:
            value = float(value)
        
        self.value = value

# errors 

class LexicError(BaseException):
    
    @classmethod
    def invalid_lexeme(cls, lexeme: str):
        return cls(f'[INVALID LEXEME: {lexeme}]')

class SyntacticError(BaseException):

    @classmethod
    def invalid_syntax(cls, expected_lexeme: str, found_lexeme: str):
        return cls(f'[EXPECTED LEXEME: {expected_lexeme} GOT LEXEME: {found_lexeme}]')    

class SemanticError(BaseException):
    
    @classmethod
    def conflicting_variable_declaration(cls, binding_name: str):
        return cls(f'[BINDING NAME: {binding_name} IS ALEADY TAKEN IN THIS SCOPE]')
    
    @classmethod
    def non_existent_variable(cls, binding_name: str):
        return cls(f'[BINDING WITH NAME: {binding_name}] DOES NOT EXIST]')

    @classmethod
    def invalid_arg_type(cls, expected_type:Union[type, str], found_type:Union[type, str]):
        return cls(f'[INVALID ARG TYPE FOR STATEMENT EXPECTED: {expected_type} GOT: {found_type}]')

# lexer for lexic analysis
# ignore a 
# white spaces tabs and new lines will be delimitators

# remove comments
class Tokenizer:
    
    def __init__(self, file_path: str):
        """ loads data from file_path """
        path = Path(file_path)
        if not path.exists():
            raise Exception('file path does not exist')
        
        with open(path, 'r') as file:
            self.raw_data = list(file.read())

        # delimitation characters
        self.delim = {' ', '\n', '\t', '\0'}

        # tokens to ignore
        self.ignore = {}
        
        # cur when reading raw data
        self.cur_index = 0

    def lexeme_matches_token(self, lexeme: str, token_type: TokenType) -> bool:
        match = re.fullmatch(token_type.value, lexeme)        
        if match == None:
            return False
        return True

    def get_token(self, lexeme: str) -> Token:
        """ Returns a token corresponding to a lexeme 
        """
        for token_type in TokenType: 
            # special tokens not meant to be matched
            if token_type == TokenType.EOF or token_type == TokenType.UNKNOWN:
                continue
            if self.lexeme_matches_token(lexeme, token_type):
                return Token(token_type, lexeme)
        
        return Token(TokenType.UNKNOWN, lexeme)

    def get_next_lexeme(self) -> Optional[str]:
        """ Reads a lexeme from the current position on the raw data
        and returns it 
        """
        lexeme = ''
        while ( self.cur_index < len(self.raw_data) and 
                self.raw_data[self.cur_index] in self.delim):
            
            self.cur_index += 1
        
        while ( self.cur_index < len(self.raw_data) and
                self.raw_data[self.cur_index] not in self.delim):
            
            lexeme += self.raw_data[self.cur_index]
            self.cur_index += 1
        
        return lexeme if len(lexeme) > 0 else None
    
    def trim_off_comments(self, tokens: List[Token]) -> List[Token]:
        final_tokens = []
        inside_comment = False

        for token in tokens:
            if token.token_type == TokenType.OPEN_COMMENT:
                inside_comment = True
            
            if inside_comment is not True:
                final_tokens.append(token)
            
            if token.token_type == TokenType.CLOSE_COMMENT and inside_comment == True:
                inside_comment = False 
            elif token.token_type == TokenType.CLOSE_COMMENT and inside_comment == False:
                raise SyntaxError("[TOKENIZATION EXCEPTION comment closing withouht opening in]")
        
        if inside_comment == True:
            raise SyntaxError("[TOKENIZATION EXCEPTION COMMENT opening withouth closing]")

        return final_tokens

    def check_for_unknown_tokens(self, tokens: List[Token]):
        """ Called once comments were removed and raises error message if it detects an unknown token
        """
        for token in tokens:
            if token.token_type == TokenType.UNKNOWN:
                raise LexicError.invalid_lexeme(str(token.value))


    def tokenize_data(self) -> List[Token]:
        """ Returns array of tokens with comments trimmed off 
        """
        cur_lexeme = self.get_next_lexeme()
        tokens = []
                
        while cur_lexeme != None:
            token = self.get_token(cur_lexeme)
            tokens.append(token)
            cur_lexeme = self.get_next_lexeme()

        eof_token = Token(TokenType.EOF, TokenType.EOF.value)
        tokens.append(eof_token)
        
        tokens = self.trim_off_comments(tokens)
        
        self.check_for_unknown_tokens(tokens)

        return tokens

#       - backtrack when found a parsing error
#       - can return the line at which it ended as an index to considereate

class ParsingResultType(Enum):
    SUCCESS = 200
    ERROR = 400

class ParsingResult:
    def __init__(self, type: ParsingResultType, 
                 msg: str, end_index: int):
        self.type = type
        self.msg = msg
        self.end_index = end_index
    
    @classmethod
    def success(cls, end_index):
        return cls(ParsingResultType.SUCCESS, 'success', end_index)
    
    @classmethod
    def error(cls):
        return cls(ParsingResultType.ERROR, 'error', 0)

# parser for syntax with recursive descent
# here execution of code will take place
class Parser:
    
    def __init__(self, tokens: List[Token]): 
    
        self.cur_index = 0
        self.tokens = tokens
        
        
        self.cur_scope = 0
        # dictionary stack representing scopes to score variables 
        self.scopes: List[Dict[str, Union[float, str]]] = [{}] 

    def get_cur_token(self) -> Token:
        cur_token = self.tokens[self.cur_index]
        self.cur_index += 1
        return cur_token
    
    # turtle script actions
    def center_turtle(self):
        turtle.penup()
        turtle.goto(0, 0)
        turtle.pendown()

    def save_variable(self, binding_name: str, value: Union[float, str]):
        if self.scopes[len(self.scopes)-1].get(binding_name) is not None: 
            raise SemanticError.conflicting_variable_declaration(binding_name)
        self.scopes[len(self.scopes)-1][binding_name] = value    
    
    # parser statements

    def fetch_variable(self, binding_name: str, expected_type: type=float) -> Union[float, str]:
        """ Returns the variable associated with binding name
        If such variable does not exist, an error is raised
        """
        cur_scope = len(self.scopes) - 1
        
        while cur_scope > -1:
            var = self.scopes[cur_scope].get(binding_name)
            if var != None:
                return var 
            cur_scope -= 1
        
        raise SemanticError.non_existent_variable(binding_name)
        
    def parse_instruction_batch(self, index) -> ParsingResult:
        """ Recursively processes a batch of instructions until the scope is 
        closed
        """
        if self.tokens[index].token_type == TokenType.CLOSE_SCOPE:
            return ParsingResult.success(index+1)
        
        line_statement_parsing_result = self.parse_line_statement(index)
        if line_statement_parsing_result.type == ParsingResultType.SUCCESS:
            index = line_statement_parsing_result.end_index
            batch_parsing_result = self.parse_instruction_batch(index)
            if batch_parsing_result.type == ParsingResultType.SUCCESS:
                return batch_parsing_result

        scope_statement_parsing_result = self.parse_scope_statement(index)
        if scope_statement_parsing_result.type == ParsingResultType.SUCCESS:
            index = scope_statement_parsing_result.end_index 
            batch_parsing_result = self.parse_instruction_batch(index)
            if batch_parsing_result.type == ParsingResultType.SUCCESS:
                return batch_parsing_result
        
        return ParsingResult.error()
        
    def parse_scoped_statements(self, index=0) -> ParsingResult:
        if self.tokens[index].token_type != TokenType.OPEN_SCOPE:
            return ParsingResult.error()
        
        index += 1
        
        # add new empty scope 
        self.scopes.append({})

        batch_parsing_result = self.parse_instruction_batch(index)
        if batch_parsing_result.type == ParsingResultType.SUCCESS:
            # scopes is closed thus destroyed
            self.scopes.pop()
            return batch_parsing_result

        return ParsingResult(ParsingResultType.ERROR, 'error', 0)
    
    def parse_loop_statement(self, index) -> ParsingResult:
        """ parses a loop and verifies that the argument given to the loop statement is 
        of correct type
        """
        if self.tokens[index].token_type != TokenType.FOR_LOOP:
            return ParsingResult.error()
        index += 1

        loop_times = None
        
        if self.tokens[index].token_type == TokenType.NUMBER_LITERAL:
            loop_times = self.tokens[index].value
        elif self.tokens[index].token_type == TokenType.BINDING_NAME:
            binding_name = self.tokens[index].value
            loop_times = self.fetch_variable(binding_name)
        
        # check type of loop times
        if type(loop_times) == str:
            raise SemanticError.invalid_arg_type(expected_type='integer number', found_type='colour')
        elif loop_times % 1 != 0:
            raise SemanticError.invalid_arg_type(expected_type='integer number', found_type='float number')
        
        loop_times = int(loop_times)
        index += 1
        end_index = 0
        
        for _ in range(loop_times):
            scoped_statements_parsing_result = self.parse_scoped_statements(index)
            if scoped_statements_parsing_result.type == ParsingResultType.ERROR:
                return scoped_statements_parsing_result
            end_index = scoped_statements_parsing_result.end_index

        return ParsingResult.success(end_index)
        
    def parse_scope_statement(self, index) -> ParsingResult:
        """ Tries to parse scope statement """
        
        scoped_statements_parse_result = self.parse_scoped_statements(index)
        if scoped_statements_parse_result.type == ParsingResultType.SUCCESS:
            return scoped_statements_parse_result
        
        loop_statement_parse_result = self.parse_loop_statement(index)
        if loop_statement_parse_result.type == ParsingResultType.SUCCESS:
            return loop_statement_parse_result
        
        return ParsingResult.error()

    def parse_arg_line_statement(self, index: int, statement_token: TokenType, 
                                 arg_token: TokenType, arg_type: type, 
                                 callback
        ) -> ParsingResult:
        """ Parses all the statements that consist of a keyword and an argument
        EX: dela 10 ;
        """

        if self.tokens[index].token_type != statement_token:
            return ParsingResult.error()
        index += 1
        value: arg_type

        if self.tokens[index].token_type == TokenType.BINDING_NAME:
            # value = self.fetch_variable()
            binding_name = self.tokens[index].value
            value = self.fetch_variable(binding_name)
            if type(value) != arg_type:
                raise BaseException('semantic error binding type does not match needed arg type')
        elif self.tokens[index].token_type == arg_token:
            value = self.tokens[index].value
        else:
            return ParsingResult.error() 
        
        if self.tokens[index+1].token_type != TokenType.DELIMITATOR:
            return ParsingResult.error()

        callback(value)

        return ParsingResult.success(index+2)

    def parse_non_arg_line_statement(self, index: int, statement_token: TokenType,
                                     callback
        ) -> ParsingResult:
        """ Parses all the statements that consist of a single command without an argument 
        EX: cntr
        """
        
        if self.tokens[index].token_type != statement_token:
            return ParsingResult.error() 
        index += 1

        if self.tokens[index].token_type != TokenType.DELIMITATOR:
            return ParsingResult.error() 
        
        callback()

        return ParsingResult.success(index+1)

    def parse_binding_definition(self, index: int, 
                                 binding_type_token: TokenType,
                                 binding_literal_token: TokenType
        ) -> ParsingResult:
        """ Parses a binding definition of type binding_type and with value value
        """
        if self.tokens[index].token_type != TokenType.LET_KEYWORD:
            return ParsingResult.error()
        index += 1

        if self.tokens[index].token_type != TokenType.BINDING_NAME:
            return ParsingResult.error() 
        
        binding_name = self.tokens[index].value
        index+=1

        if self.tokens[index].token_type != TokenType.COLON:
            return ParsingResult.error()
        index += 1

        if self.tokens[index].token_type != binding_type_token:
            return ParsingResult.error()
        index += 1

        if self.tokens[index].token_type != TokenType.EQUAL_OPERATOR:
            return ParsingResult.error()
        index += 1

        if self.tokens[index].token_type != binding_literal_token:
            return ParsingResult.error()
        value = self.tokens[index].value
        index += 1

        if self.tokens[index].token_type != TokenType.DELIMITATOR:
            return ParsingResult.error()
        index += 1

        # save binding 
        self.save_variable(binding_name, value)

        return ParsingResult.success(index)

    def parse_line_statement(self, index=0) -> ParsingResult:
        # single arged line statement

        forward_parse_result = self.parse_arg_line_statement(index, TokenType.FORWARD_MOVE, 
                                                             TokenType.NUMBER_LITERAL, float, 
                                                             turtle.forward)
        if forward_parse_result.type == ParsingResultType.SUCCESS:
            return forward_parse_result
    
        backward_parse_result = self.parse_arg_line_statement(index, TokenType.BACKWARDS_MOVE, 
                                                              TokenType.NUMBER_LITERAL, float, 
                                                              turtle.backward)
        if backward_parse_result.type == ParsingResultType.SUCCESS:
            return backward_parse_result
        
        left_turn_parse_result = self.parse_arg_line_statement(index, TokenType.LEFT_TURN, 
                                                               TokenType.NUMBER_LITERAL, float, 
                                                               turtle.left)
        if left_turn_parse_result.type == ParsingResultType.SUCCESS:
            return left_turn_parse_result

        right_turn_parse_result = self.parse_arg_line_statement(index, TokenType.RIGHT_TURN, 
                                                                TokenType.NUMBER_LITERAL, float, 
                                                                turtle.right)
        if right_turn_parse_result.type == ParsingResultType.SUCCESS:
            return right_turn_parse_result      
        
        change_colour_parse_result = self.parse_arg_line_statement(index, TokenType.CHANGE_COLOUR,
                                                                   TokenType.COLOUR_LITERAL, str, 
                                                                   turtle.pencolor)
        if change_colour_parse_result.type == ParsingResultType.SUCCESS:
            return change_colour_parse_result

        # non-arged line statements

        pen_up_parse_result = self.parse_non_arg_line_statement(index, TokenType.PEN_UP, 
                                                                turtle.penup)
        if pen_up_parse_result.type == ParsingResultType.SUCCESS:
            return pen_up_parse_result 
        
        pen_down_parse_result = self.parse_non_arg_line_statement(index, TokenType.PEN_DOWN, 
                                                                  turtle.pendown)
        if pen_down_parse_result.type == ParsingResultType.SUCCESS:
            return pen_down_parse_result
        
        center_parse_result = self.parse_non_arg_line_statement(index, TokenType.CENTER,
                                                                lambda: turtle.setposition(0, 0))
        if center_parse_result.type == ParsingResultType.SUCCESS:
            return center_parse_result 
        
        clear_parse_result = self.parse_non_arg_line_statement(index, TokenType.CLEAR, 
                                                               self.center_turtle)
        if clear_parse_result.type == ParsingResultType.SUCCESS:
            return clear_parse_result
    
        # binding definitions
        number_binding_parse_result = self.parse_binding_definition(index, TokenType.NUMBER_DATA_TYPE, 
                                                                    TokenType.NUMBER_LITERAL)
        if number_binding_parse_result.type == ParsingResultType.SUCCESS:
            return number_binding_parse_result    

        colour_binding_parse_result = self.parse_binding_definition(index, TokenType.COLOUR_DATA_TYPE, 
                                                                    TokenType.COLOUR_LITERAL)
        if colour_binding_parse_result.type == ParsingResultType.SUCCESS:
            return colour_binding_parse_result
        
        return ParsingResult(ParsingResultType.ERROR, 'ERROR', 0)

    def parse_statement(self, index=0) -> ParsingResult:
        line_statement_parsing_result = self.parse_line_statement(index)
        if line_statement_parsing_result.type == ParsingResultType.SUCCESS:
            return line_statement_parsing_result
        
        scope_statement_parsing_result = self.parse_scope_statement(index)
        if scope_statement_parsing_result.type == ParsingResultType.SUCCESS:
            return scope_statement_parsing_result
                
        return ParsingResult(ParsingResultType.ERROR, 'ERROR', 0)


    # parses with recursive descent method the 
    # stream of tokens and executes it
    def parse_program(self, index=0) -> ParsingResult:
        if self.tokens[index].token_type == TokenType.EOF:
            return ParsingResult.success(index)
        
        statement_parsing_result = self.parse_statement(index)
        if statement_parsing_result.type == ParsingResultType.ERROR:
            return statement_parsing_result
        
        index = statement_parsing_result.end_index

        program_parsing_result = self.parse_program(index)
        if program_parsing_result.type == ParsingResultType.ERROR:
            return program_parsing_result 

        return ParsingResult.success(program_parsing_result.end_index)

def main(args):
    file_name = args[1]
    
    tokenizer = Tokenizer(file_name)
    tokens = tokenizer.tokenize_data()
    print('TOKENS OF THE PROGRAM: ', list(map(lambda token: token.value, tokens)))

    parser = Parser(tokens)

    turtle.begin_fill()
    parsing_result = parser.parse_program()
    turtle.end_fill()
    turtle.done()

    print(parser.scopes)
    print(parsing_result.type)


if __name__ == '__main__':
    main(sys.argv)
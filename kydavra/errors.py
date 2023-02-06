'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vladimir.stojoc@gmail.com
'''
class NotBetweenZeroAndOneError(BaseException):
    ''' Raised when a value is not between zero and one, but it should be'''
    pass

class NotBinaryData(BaseException):
    ''' Raised when the data passed is not binary '''
    pass

class NoSuchMethodError(BaseException):
    ''' Raised when the selector or reducer doesnt't have a method '''
    pass

class MissingDataError(BaseException):
    ''' Raised when the dataframe is missing values'''
    pass


class NonNumericDataError(BaseException):
    ''' Raised when the dataframe has non numeric values'''
    pass


class NoSuchColumnError(BaseException):
    ''' Raised when the target column is missing or written wrong'''
    pass

class DifferentColumnsError(BaseException):
    ''' The passed dataframe has different columns from the one passed to filter function '''
    pass
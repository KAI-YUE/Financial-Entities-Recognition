"""
Read configurations from config.json.
"""
# Python Libraries
import json

class DictClass(object):
    """
    Turns a dictionary into a class
    """
 
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
 
    def __repr__(self):
        """"""
        return "<DictClass: {}>".format(self.__dict__)
    
def loadConfig(file_name):
    with open(file_name, "r") as fp:
        config = json.load(fp)
    
    return DictClass(config) 
    
if __name__ == '__main__':
    test = loadConfig()
    
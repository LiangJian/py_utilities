import fileinput


# class for input
class Params:
    param = dict()
    input_file_name = ''

    @classmethod
    def __init__(cls, input_file):
        cls.input_file_name = input_file

    @classmethod
    def add(cls, name, ini_value):
        cls.param[name] = ini_value

    @classmethod
    def parse_input(cls):
        print('Parsing inputs ...')
        for line in fileinput.input(cls.input_file_name):
            name_ = line.split()[0]
            if name_[0] == '#':
                continue
            if name_ in cls.param:
                if type(cls.param[name_]) == bool:
                    if line.split()[1] == 'True':
                        cls.param[name_] = True
                    else:
                        if line.split()[1] == 'False':
                            cls.param[name_] = False
                        else:
                            print('wrong value for bool')
                            exit(-1)
                else:
                    cls.param[name_] = type(cls.param[name_])(line.split()[1])
                print("%-25s" % name_, "%-4s" % '=', line.split()[1])
        print('Parsing done.')

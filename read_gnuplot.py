import numpy as np
########################################


def read(filename, a):
    a.clear()
    lines = []
    f = None
    try:
        f = open(filename, 'r')
        lines.append(f.readlines())
    except IOError as e:  
        print(e)
    finally:  
        f.close()

    count_blank = 0
    count_block = 0
    count_line = 0
    for i in range(0, len(lines[0])):
        arr = lines[0][i].split()
        if i == 0:
            a.append([])  # new block
            # print(count_block)
        if len(arr) == 0 and count_blank == 0:
            count_blank += 1
            continue
        else:
            if len(arr) == 0 and count_blank == 1:
                count_blank += 1
                a.append([])  # new block
                count_block += 1
                count_line = 0
                # print(count_block)
                continue
            else:
                if len(arr) == 0 and count_blank >= 2:
                    continue
                else:
                    if len(arr) != 0 and str(arr[0][0]) == '#':
                        count_blank = 0
                        continue
                    else:
                        count_blank = 0
                        a[count_block].append([])  # new line
                        # print(count_block,count_line)
                        for j in range(0, len(arr)):
                            try:
                                a[count_block][count_line].append(float(arr[j]))
                            except ValueError:
                                a[count_block][count_line].append(0.0)
                        count_line += 1


########################################


def pick(a, d, offset, column):
    d.clear()
    for i in range(0, len(column)):
        d.append(np.array(a[offset])[0:, column[i]])

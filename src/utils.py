def write_result(src, dst):
    result = []
    with open(src, 'r') as fp:
        test_data = fp.readlines()
    #    
    i = -1
    j = -1
    with open(dst, 'r') as fp:
        for predict_data in fp:
            if predict_data[0] == '[':
                if predict_data[1] == 'C':
                    result.append([[], [], []])     # Text, predicted label, ground Truth
                    i += 1
                elif predict_data[1] == 'S':
                    while test_data[j+1] != '\n':
                        j += 1
                    j += 1
                    result[i][0] = "".join(result[i][0])
                    result[i][1] = "".join(result[i][1])
                    result[i][2] = "".join(result[i][2])

            else:
                j += 1
                if predict_data[0] == 'B':
                    if result[i][1] == []:
                        pass
                    else:
                        result[i][1].append(',')
                    result[i][1].append(test_data[j][0])
                elif predict_data[0] == 'I':
                    result[i][1].append(test_data[j][0])
                
                result[i][0].append(test_data[j][0])
                
                if len(test_data[j]) >= 3:
                    if test_data[j][2] == 'B':
                        if result[i][2] == []:
                            pass
                        else:
                            result[i][2].append(',')
                        
                        result[i][2].append(test_data[j][0])
                    
                    elif test_data[j][2] == 'I':
                        result[i][2].append(test_data[j][0])
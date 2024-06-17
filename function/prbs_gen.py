import numpy as np
def generate_prbs(pseudo_random_state, init_value=None, expression=None,PAM=2,numofbits=None):

    if pseudo_random_state == 'user_define':
        pseudo_random_sequence = real_calculate_prbs(init_value, expression,PAM,numofbits=None)
    else:
        pseudo_random_dict = {'prbs_7': ['1111101', [7, 3],PAM],
                              'prbs_9': ['111110101', [9, 4],PAM],
                              'prbs_15': ['111110101101110', [15, 1],PAM],
                              'prbs_16': ['1111101011011100', [16, 12, 3, 1],PAM],
                              'prbs_20': ['11111010110111001011', [20, 3],PAM],
                              'prbs_21': ['111110101101110010111', [21, 2],PAM],
                              'prbs_23': ['11111010110111001011101', [23, 5],PAM],
                              'prbs_31':['1000000000000000000000000000001',[31,28],PAM]}
        pseudo_random_sequence = real_calculate_prbs(pseudo_random_dict[pseudo_random_state][0],
                                                     pseudo_random_dict[pseudo_random_state][1],PAM,numofbits)
    return pseudo_random_sequence


# 真正计算伪随机序列

# 用xrange省空间同时提高效率
def real_calculate_prbs(value, expression,PAM,numofbits=None):

    #将字符串转化为列表
    value_list = [int(i) for i in list(value)]
    #计算伪随机序列周期长度
    if numofbits ==None:
        pseudo_random_length = (2 << (len(value) - 1))-1
    else:
        pseudo_random_length = int(numofbits)

    sequence = []


    #产生规定长度的伪随机序列
    for i in range(pseudo_random_length):

        mod_two_add = sum([value_list[t-1] for t in expression])
        xor = mod_two_add % 2

        #计算并得到伪随机序列的状态值
        value_list.insert(0, xor)


        #sequence.append(value_list[-1])
        sequence.append(xor)
        #del value_list[-1]
    if PAM ==2:
        return sequence
    elif PAM ==4:
        s = np.append(sequence,sequence)
        pam_sequence = np.zeros(np.size(sequence))
        for i in range(np.size(sequence)):
            if(s[2*i]==0 and s[2*i+1]==0):
                pam_sequence[i]=-1
            elif(s[2*i]==0 and s[2*i+1]==1):
                pam_sequence[i]=-0.333
            elif(s[2*i]==1 and s[2*i+1]==1):
                pam_sequence[i]=0.333
            else:
                pam_sequence[i]=1
        return  pam_sequence



#if __name__ == '__main__':

#    result_data = generate_prbs('prbs_31',PAM=4,numofbits=31)

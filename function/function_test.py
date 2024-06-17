def rearrange_list(N):
    # 创建一个从0到2N-1的列表
    lst = list(range(4*N))

    # 创建一个新的列表来存储重新排列后的元素
    new_lst = [None] * (4*N)

    # 按照指定的顺序填充新的列表
    for i in range(N):
        new_lst[4*i] = lst[2*i]
        new_lst[4*i+1] = lst[2*i+1]
        new_lst[4*i+2] = lst[2*i+2*N]
        new_lst[4*i+3] = lst[2*i+1+2*N]

    return new_lst

# 测试函数
N = 12
print(rearrange_list(int(N/4)))